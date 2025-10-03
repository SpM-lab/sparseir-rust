#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== Eigen3 Jacobi SVD Test ===" << std::endl;
    
    // Test case: 2x2 rank-one matrix
    Eigen::Matrix2d A;
    A << 1.0, 1.0,
         1.0, 1.0;
    
    std::cout << "Input matrix A:" << std::endl;
    std::cout << A << std::endl;
    
    // Compute SVD using Eigen3
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    std::cout << "\nSingular values: " << svd.singularValues().transpose() << std::endl;
    std::cout << "U matrix:" << std::endl << svd.matrixU() << std::endl;
    std::cout << "V matrix:" << std::endl << svd.matrixV() << std::endl;
    
    // Check reconstruction
    Eigen::Matrix2d reconstructed = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
    std::cout << "\nReconstruction U * S * V^T:" << std::endl;
    std::cout << reconstructed << std::endl;
    
    std::cout << "\nReconstruction error:" << std::endl;
    std::cout << (A - reconstructed) << std::endl;
    
    std::cout << "\nMax error: " << (A - reconstructed).cwiseAbs().maxCoeff() << std::endl;
    
    return 0;
}
