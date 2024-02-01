from tensor import TensorSpec, Tensor
from algorithm import vectorize
from sys.info import simdwidthof
from math import min, max
from memory import memset_zero


struct Matrix2D[dtype : DType]:
    var height: Int
    var width: Int
    var data: DTypePointer[dtype]

    fn __init__(inout self, height: Int, width: Int):
        self.height = height
        self.width = width
        self.data = DTypePointer[dtype].alloc(height * width)
    
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


struct Kernel2D[dtype : DType]:
    var width: Int
    var weights: Matrix2D[dtype]

    fn __init__(inout self, width: Int):
        debug_assert(width % 2 == 1, "`width` must be odd")
        self.width = width
        self.weights = Matrix2D[dtype](width, width)
    
    fn apply(self, input_channel: Matrix2D[dtype], inout output_channel: Matrix2D[dtype]):
        input_channel.assert_same_dims(output_channel)
        var sample = Matrix2D[dtype](self.width, self.width)
        
        for focus_row in range(input_channel.height):
            let unsafe_start_row = focus_row - self.width//2

            for focus_col in range(input_channel.width):
                let unsafe_start_col = focus_col - self.width//2
                let unsafe_end_col = focus_col + self.width//2 + 1
                let start_col = max(unsafe_start_col, 0)
                let end_col = min(unsafe_end_col, input_channel.width)
                let n_cols = end_col - start_col
                let kernel_col = start_col - unsafe_start_col

                sample.clean()

                for kernel_row in range(self.width):
                    let src_row = unsafe_start_row + kernel_row
                    if not 0 <= src_row < input_channel.height:
                        continue
                    
                    sample.cp_segment_from(input_channel, src_row, start_col, kernel_row, kernel_col, n_cols)
                
                output_channel[focus_row, focus_col] += self.weights.dot(sample)
        
    
    fn to_tensor(self) -> Tensor[dtype]:
        return self.weights.to_tensor()


fn main():
    alias dtype = DType.float32

    var img = Matrix2D[dtype](5, 5)
    for i in range(img.height):
        for j in range(img.width):
            img[i, j] = i * img.width + j + 1
    
    var kernel = Kernel2D[dtype](3)
    for i in range(kernel.width):
        for j in range(kernel.width):
            kernel.weights[i, j] = 1 * (i * img.width + j)
    
    var out = Matrix2D[dtype](5, 5)
    out.clean()
    
    kernel.apply(img, out)
    print(out.to_tensor())
