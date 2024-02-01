from matrix2d import Matrix2D
from kernel2d import Kernel2D


fn sample_matrix[dtype:DType](height: Int, width: Int) -> Matrix2D[dtype]:
    var img = Matrix2D[dtype](height, width)
    for i in range(img.height):
        for j in range(img.width):
            img[i, j] = i * img.width + j + 1
    return img


fn sample_kernel[dtype:DType](width: Int) -> Kernel2D[dtype]:
    var kernel = Kernel2D[dtype](width)
    let weights = sample_matrix[dtype](width, width)
    kernel.set_weights(weights)
    return kernel


fn main():
    alias dtype = DType.float16

    var img = sample_matrix[dtype](5, 5)
    
    var kernel = sample_kernel[dtype](3)
    
    var out = img.same_dim_zeros()
    
    kernel.apply(img, out)
    
    print(out.to_tensor())
