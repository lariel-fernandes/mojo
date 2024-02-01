from .matrix2d import MultiChannelMatrix2D
from algorithm import parallelize


@register_passable("trivial")
struct Conv2D[dtype: DType]:
    var num_out_feats: Int
    var kernels: Pointer[MultiChannelKernel2D[dtype]]

    fn __init__(num_in_feats: Int, num_out_feats: Int, kernel_width: Int) -> Self:
        debug_assert(num_in_feats >= 1, "`num_in_feats` must be greater than 0")
        debug_assert(num_out_feats >= 1, "`num_out_feats` must be greater than 0")
        var kernels = Pointer[MultiChannelKernel2D[dtype]].alloc(num_out_feats)
        for i in range(num_out_feats):
            kernels[i] = MultiChannelKernel2D[dtype](num_in_feats, kernel_width)
        return Conv2D[dtype] {num_out_feats: num_out_feats, kernels: kernels}

    fn __init__(*kernels: MultiChannelKernel2D[dtype]) -> Self:
        debug_assert(len(kernels) >= 1, "`num_kernels` must be greater than 0")
        var idx = 0
        var ptr = Pointer[MultiChannelKernel2D[dtype]].alloc(len(kernels))
        let first = kernels[0]
        for idx in range(len(kernels)):
            let kernel = kernels[idx]
            kernel.weights.assert_same_dims(first.weights)
            ptr[idx] = kernel
        return Conv2D[dtype] {num_out_feats: len(kernels), kernels: ptr}
    
    @always_inline
    fn num_in_feats(self) -> Int:
        return self.kernels[0].num_channels()
    
    @always_inline
    fn kernel_width(self) -> Int:
        return self.kernels[0].width()
    
    fn apply(self, input: MultiChannelMatrix2D[dtype], inout output: MultiChannelMatrix2D[dtype]):
        input.assert_same_dims(output)

        @parameter
        fn apply(idx: Int):
            self.kernels[idx].apply(input, output[idx])
        
        parallelize[apply](self.num_out_feats)
    
    fn output_buffer_for(self, input: MultiChannelMatrix2D[dtype]) -> MultiChannelMatrix2D[dtype]:
        return MultiChannelMatrix2D[dtype].zeros(self.num_out_feats, input.height(), input.width())
