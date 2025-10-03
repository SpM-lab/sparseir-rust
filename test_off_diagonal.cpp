#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== Eigen3 Off-Diagonal Matrix Test ===" << std::endl;
    
    Eigen::MatrixXd A(2, 2);
    A << 0, 1,
         1, 0;
    
    std::cout << "Input matrix A:\n" << A << std::endl;
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    std::cout << "\nSingular values: " << svd.singularValues().transpose() << std::endl;
    std::cout << "U matrix:\n" << svd.matrixU() << std::endl;
    std::cout << "V matrix:\n" << svd.matrixV() << std::endl;
    
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    S.diagonal() = svd.singularValues();
    
    Eigen::MatrixXd reconstructed = svd.matrixU() * S * svd.matrixV().transpose();
    std::cout << "\nReconstruction U * S * V^T:\n" << reconstructed << std::endl;
    
    Eigen::MatrixXd error = A - reconstructed;
    std::cout << "\nReconstruction error:\n" << error << std::endl;
    std::cout << "Max error: " << error.norm() << std::endl;
    
    return 0;
}
