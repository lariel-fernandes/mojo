from tensor import TensorSpec, Tensor
from algorithm import vectorize
from sys.info import simdwidthof
from memory import memset_zero


struct Matrix2D[dtype : DType]:
    var height: Int
    var width: Int
    var data: DTypePointer[dtype]

    fn __init__(inout self, height: Int, width: Int):
        self.height = height
        self.width = width
        self.data = DTypePointer[dtype].alloc(height * width)
    
    fn __copyinit__(inout self, existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.data = DTypePointer[dtype].alloc(existing._size())
        memcpy[dtype](self.data, existing.data, existing._size())
    
    fn _yx_to_offset(self, row: Int, col: Int) -> Int:
        return row * self.width + col
    
    fn _size(self) -> Int:
        return self.height * self.width
    
    fn load[n:Int](self, row: Int, col: Int) -> SIMD[dtype, n]:
        return self.data.simd_load[n](self._yx_to_offset(row, col))
    
    fn store[n:Int](self, row: Int, col: Int, val: SIMD[dtype, n]):
        self.data.simd_store[n](self._yx_to_offset(row, col), val)
    
    fn __getitem__(inout self, row: Int, col: Int) -> SIMD[dtype, 1]:
        return self.load[1](row, col)
    
    fn __setitem__(inout self, row: Int, col: Int, val: SIMD[dtype, 1]):
        self.store[1](row, col, val)
    
    fn clean(inout self):
        memset_zero[dtype](self.data, self._size())
    
    fn assert_same_dims(self, other: Matrix2D[dtype]) -> None:
        debug_assert(self.width == other.width, "`width` must match")
        debug_assert(self.height == other.height, "`height` must match")
    
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
    
    fn cp_all(inout self, other: Matrix2D[dtype]):
        self.assert_same_dims(other)
        self.cp_segment_from(other, 0, 0, 0, 0, self._size())
    
    fn same_dim_zeros(self) -> Matrix2D[dtype]:
        var out = Matrix2D[dtype](self.height, self.width)
        out.clean()
        return out
