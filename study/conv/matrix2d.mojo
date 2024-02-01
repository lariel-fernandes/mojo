from tensor import TensorSpec, Tensor
from algorithm import vectorize
from sys.info import simdwidthof
from memory import memset_zero


@register_passable("trivial")
struct Matrix2D[dtype : DType]:
    var height: Int
    var width: Int
    var data: DTypePointer[dtype]

    @always_inline
    fn __init__(height: Int, width: Int) -> Self:
        let data = DTypePointer[dtype].alloc(height * width)
        return Matrix2D[dtype] {height: height, width: width, data: data}
    
    @always_inline
    fn clean(inout self):
        memset_zero[dtype](self.data, self._size())
    
    @always_inline
    @staticmethod
    fn zeros(height: Int, width: Int) -> Matrix2D[dtype]:
        var out = Matrix2D[dtype](height, width)
        out.clean()
        return out
    
    @always_inline
    fn same_dim_zeros(self) -> Matrix2D[dtype]:
        return Matrix2D[dtype].zeros(self.height, self.width)
    
    @always_inline
    fn _yx_to_offset(self, row: Int, col: Int) -> Int:
        return row * self.width + col
    
    @always_inline
    fn _size(self) -> Int:
        return self.height * self.width
    
    @always_inline
    fn load[n:Int](self, row: Int, col: Int) -> SIMD[dtype, n]:
        return self.data.simd_load[n](self._yx_to_offset(row, col))
    
    @always_inline
    fn store[n:Int](self, row: Int, col: Int, val: SIMD[dtype, n]):
        self.data.simd_store[n](self._yx_to_offset(row, col), val)
    
    @always_inline
    fn __getitem__(inout self, row: Int, col: Int) -> SIMD[dtype, 1]:
        return self.load[1](row, col)
    
    @always_inline
    fn __setitem__(inout self, row: Int, col: Int, val: SIMD[dtype, 1]):
        self.store[1](row, col, val)
    
    @always_inline
    fn assert_same_dims(self, other: Matrix2D[dtype]) -> None:
        debug_assert(self.width == other.width, "`width` must match")
        debug_assert(self.height == other.height, "`height` must match")
    
    @always_inline
    fn assert_square(self) -> None:
        debug_assert(self.height == self.width, "`width` and `height` must match")
    
    @always_inline
    fn assert_centered(self) -> None:
        self.assert_square()
        debug_assert(self.width % 2 == 1, "`width` must be odd")
    
    fn to_tensor(self) -> Tensor[dtype]:
        let spec = TensorSpec(dtype, self.height, self.width)
        var ptr = DTypePointer[dtype].alloc(self._size())
        memcpy[dtype](ptr, self.data, self._size())
        return Tensor[dtype](ptr, spec)
    
    fn dot(self, other: Matrix2D[dtype]) -> SIMD[dtype, 1]:
        alias vec_size = simdwidthof[dtype]()
        self.assert_same_dims(other)
        var acc = SIMD[dtype, 1]()

        @parameter
        fn apply[n:Int](offset: Int) capturing -> None:
            let a = self.data.simd_load[n](offset)
            let b = other.data.simd_load[n](offset)
            acc += (a * b).reduce_add()
        
        vectorize[vec_size, apply](self.width * self.height)
        return acc
    
    fn cp_segment_from(inout self, other: Matrix2D[dtype], src_row: Int, src_col: Int, tgt_row: Int, tgt_col:Int, length: Int):
        let tgt = self.data.offset(tgt_row * self.width + tgt_col)
        let src = other.data.offset(src_row * other.width + src_col)
        memcpy[dtype](tgt, src, length)


@register_passable("trivial")
struct MultiChannelMatrix2D[dtype: DType]:
    var num_channels: Int
    var channels: Pointer[Matrix2D[dtype]]

    fn __init__(num_channels: Int, height: Int, width: Int) -> Self:
        debug_assert(num_channels >= 1, "`num_channels` must be greater than 0")
        var channels = Pointer[Matrix2D[dtype]].alloc(num_channels)
        for i in range(num_channels):
            channels[i] = Matrix2D[dtype](height, width)
        return MultiChannelMatrix2D[dtype] {num_channels: num_channels, channels: channels}

    fn __init__(*channels: Matrix2D[dtype]) -> Self:
        debug_assert(len(channels) >= 1, "`num_channels` must be greater than 0")
        var idx = 0
        var ptr = Pointer[Matrix2D[dtype]].alloc(len(channels))
        let first = channels[0]
        for idx in range(len(channels)):
            let ch = channels[idx]
            ch.assert_same_dims(first)
            ptr[idx] = ch
        return MultiChannelMatrix2D[dtype] {num_channels: len(channels), channels: ptr}
    
    @always_inline
    fn clean(inout self):
        for idx in range(self.num_channels):
            self.channels[idx].clean()
    
    @always_inline
    @staticmethod
    fn zeros(num_channels: Int, height: Int, width: Int) -> MultiChannelMatrix2D[dtype]:
        var out = MultiChannelMatrix2D[dtype](num_channels, height, width)
        out.clean()
        return out
    
    fn same_dim_zeros(self) -> MultiChannelMatrix2D[dtype]:
        return MultiChannelMatrix2D[dtype].zeros(self.num_channels, self.height(), self.width())
    
    @always_inline
    fn __getitem__(self, index: Int) -> Matrix2D[dtype]:
        return self.channels[index]
    
    @always_inline
    fn __setitem__(self, index: Int, val: Matrix2D[dtype]):
        self.channels[index] = val
    
    @always_inline
    fn _size(self) -> Int:
        return self.num_channels * self._channel_size()
    
    @always_inline
    fn first(self) -> Matrix2D[dtype]:
        return self.channels[0]

    @always_inline
    fn assert_centered(self):
        self.first().assert_centered()
    
    @always_inline
    fn height(self) -> Int:
        return self.first().height
    
    @always_inline
    fn width(self) -> Int:
        return self.first().width
    
    @always_inline
    fn _channel_size(self) -> Int:
        return self.height() * self.width()
    
    @always_inline
    fn assert_same_dims(self, other: MultiChannelMatrix2D[dtype]) -> None:
        debug_assert(self.width() == other.width(), "`width` must match")
        debug_assert(self.height() == other.height(), "`height` must match")
        debug_assert(self.num_channels == other.num_channels, "`num_channels` must match")
    
    @always_inline
    fn assert_same_dims(self, other: Matrix2D[dtype]) -> None:
        debug_assert(self.width() == other.width, "`width` must match")
        debug_assert(self.height() == other.height, "`height` must match")

    fn to_tensor(self) -> Tensor[dtype]:
        let spec = TensorSpec(dtype, self.num_channels, self.height(), self.width())
        var ptr = DTypePointer[dtype].alloc(self._size())
        let ch_size = self._channel_size()
        for idx in range(self.num_channels):
            let ch = self.channels[idx]
            let tgt = ptr.offset(ch_size * idx)
            memcpy[dtype](tgt, ch.data, ch_size)
        return Tensor[dtype](ptr, spec)
