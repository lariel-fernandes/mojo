from tensor import Tensor
from math import min, max
from .matrix2d import Matrix2D, MultiChannelMatrix2D
from algorithm import parallelize


@register_passable("trivial")
struct Kernel2D[dtype : DType]:
    var weights: Matrix2D[dtype]

    fn __init__(width: Int) -> Self:
        let weights = Matrix2D[dtype](width, width)
        return Kernel2D[dtype](weights)
    
    fn __init__(weights: Matrix2D[dtype]) -> Self:
        weights.assert_centered()
        return Kernel2D[dtype] {weights: weights}
    
    fn apply(self, input_channel: Matrix2D[dtype], inout output_channel: Matrix2D[dtype]):
        apply_kernel(self.weights, input_channel, output_channel)
    
    @always_inline
    fn to_tensor(self) -> Tensor[dtype]:
        return self.weights.to_tensor()
    
    @always_inline
    fn width(self) -> Int:
        return self.weights.width


@register_passable("trivial")
struct MultiChannelKernel2D[dtype: DType]:
    var weights: MultiChannelMatrix2D[dtype]

    fn __init__(num_channels: Int, width: Int) -> Self:
        let weights = MultiChannelMatrix2D[dtype](num_channels, width, width)
        return MultiChannelKernel2D[dtype](weights)

    fn __init__(weights: MultiChannelMatrix2D[dtype]) -> Self:
        weights.assert_centered()
        return MultiChannelKernel2D[dtype] {weights: weights}
    
    fn apply(self, input: MultiChannelMatrix2D[dtype], inout output: Matrix2D[dtype]):
        input.assert_same_dims(output)

        @parameter
        fn apply(idx: Int):
            apply_kernel(self.weights[idx], input[idx], output)
        
        parallelize[apply](self.weights.num_channels)
    
    @always_inline
    fn to_tensor(self) -> Tensor[dtype]:
        return self.weights.to_tensor()
    
    @always_inline
    fn width(self) -> Int:
        return self.weights.width()
    
    @always_inline
    fn num_channels(self) -> Int:
        return self.weights.num_channels


fn apply_kernel[dtype: DType](weights: Matrix2D[dtype], input_channel: Matrix2D[dtype], inout output_channel: Matrix2D[dtype]):
    weights.assert_square()
    input_channel.assert_same_dims(output_channel)
    var sample = weights.same_dim_zeros()
    
    for focus_row in range(input_channel.height):
        let unsafe_start_row = focus_row - weights.width//2

        for focus_col in range(input_channel.width):
            let unsafe_start_col = focus_col - weights.width//2
            let unsafe_end_col = focus_col + weights.width//2 + 1
            let start_col = max(unsafe_start_col, 0)
            let end_col = min(unsafe_end_col, input_channel.width)
            let n_cols = end_col - start_col
            let kernel_col = start_col - unsafe_start_col

            sample.clean()

            for kernel_row in range(weights.width):
                let src_row = unsafe_start_row + kernel_row
                if not 0 <= src_row < input_channel.height:
                    continue
                
                sample.cp_segment_from(input_channel, src_row, start_col, kernel_row, kernel_col, n_cols)
            
            output_channel[focus_row, focus_col] += weights.dot(sample)
