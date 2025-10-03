#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== Eigen3 Rectangular Matrix Test ===" << std::endl;
    
    Eigen::MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;
    
    std::cout << "Input matrix A (3x2):\n" << A << std::endl;
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    std::cout << "\nSingular values: " << svd.singularValues().transpose() << std::endl;
    
    std::cout << "\nU matrix (" << svd.matrixU().rows() << "x" << svd.matrixU().cols() << "):\n" 
              << svd.matrixU() << std::endl;
    
    std::cout << "\nV matrix (" << svd.matrixV().rows() << "x" << svd.matrixV().cols() << "):\n" 
              << svd.matrixV() << std::endl;
    
    // Reconstruction
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    S.diagonal() = svd.singularValues();
    
    std::cout << "\nS matrix (" << S.rows() << "x" << S.cols() << "):\n" << S << std::endl;
    
    Eigen::MatrixXd reconstructed = svd.matrixU() * S * svd.matrixV().transpose();
    std::cout << "\nReconstruction U * S * V^T:\n" << reconstructed << std::endl;
    
    Eigen::MatrixXd error = A - reconstructed;
    std::cout << "\nReconstruction error:\n" << error << std::endl;
    std::cout << "Max error: " << error.norm() << std::endl;
    
    return 0;
}

