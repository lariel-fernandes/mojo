from tensor import Tensor
from math import min, max
from matrix2d import Matrix2D


struct Kernel2D[dtype : DType]:
    var width: Int
    var weights: Matrix2D[dtype]

    fn __init__(inout self, width: Int):
        debug_assert(width % 2 == 1, "`width` must be odd")
        self.width = width
        self.weights = Matrix2D[dtype](width, width)
    
    fn __copyinit__(inout self, existing: Self):
        self.width = existing.width
        self.weights.__copyinit__(existing.weights)
    
    fn set_weights(inout self, weights: Matrix2D[dtype]):
        self.weights.cp_all(weights)
    
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
