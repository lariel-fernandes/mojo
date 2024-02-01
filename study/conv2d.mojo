from conv import Matrix2D, MultiChannelMatrix2D, MultiChannelKernel2D, Conv2D


fn sample_matrix[dtype:DType](height: Int, width: Int) -> Matrix2D[dtype]:
    var img = Matrix2D[dtype](height, width)
    for i in range(img.height):
        for j in range(img.width):
            img[i, j] = i * img.width + j + 1
    return img


fn main():
    alias dtype = DType.float16

    let img = MultiChannelMatrix2D[dtype](
        sample_matrix[dtype](5, 5),
        sample_matrix[dtype](5, 5),
        sample_matrix[dtype](5, 5),
    )

    let conv2d = Conv2D(
        MultiChannelKernel2D[dtype](
            weights=MultiChannelMatrix2D[dtype](
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
            )
        ),
        MultiChannelKernel2D[dtype](
            weights=MultiChannelMatrix2D[dtype](
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
            )
        ),
        MultiChannelKernel2D[dtype](
            weights=MultiChannelMatrix2D[dtype](
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
            )
        ),
        MultiChannelKernel2D[dtype](
            weights=MultiChannelMatrix2D[dtype](
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
                sample_matrix[dtype](3, 3),
            )
        ),
    )

    var output_buffer = conv2d.output_buffer_for(img)

    conv2d.apply(img, output_buffer)

    print(output_buffer.to_tensor())
