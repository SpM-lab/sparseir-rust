#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== Eigen3 Rectangular Matrix (ComputeThinU/V) ===" << std::endl;
    
    Eigen::MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;
    
    std::cout << "Input matrix A (3x2):\n" << A << std::endl;
    
    // Use ComputeThinU and ComputeThinV (default)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    std::cout << "\nSingular values: " << svd.singularValues().transpose() << std::endl;
    
    std::cout << "\nU matrix (" << svd.matrixU().rows() << "x" << svd.matrixU().cols() << "):\n" 
              << svd.matrixU() << std::endl;
    
    std::cout << "\nV matrix (" << svd.matrixV().rows() << "x" << svd.matrixV().cols() << "):\n" 
              << svd.matrixV() << std::endl;
    
    // Reconstruction with thin SVD
    // S is k x k, so we use it directly
    Eigen::MatrixXd S = svd.singularValues().asDiagonal();
    
    std::cout << "\nS matrix (" << S.rows() << "x" << S.cols() << "):\n" << S << std::endl;
    
    Eigen::MatrixXd reconstructed = svd.matrixU() * S * svd.matrixV().transpose();
    std::cout << "\nReconstruction U * S * V^T (" << reconstructed.rows() << "x" << reconstructed.cols() << "):\n" 
              << reconstructed << std::endl;
    
    Eigen::MatrixXd error = A - reconstructed;
    std::cout << "\nReconstruction error:\n" << error << std::endl;
    std::cout << "Max error: " << error.norm() << std::endl;
    
    return 0;
}

