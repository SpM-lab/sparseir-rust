#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== Eigen3 Right Rotation Test ===" << std::endl;
    
    // Test matrix after left rotation: [[1.414, 0], [0, 0]]
    Eigen::MatrixXd A(2, 2);
    A << 1.414213562373095, 0.0,
         0.0, 0.0;
    
    std::cout << "Initial matrix A:\n" << A << std::endl;
    
    // Create Jacobi rotation: c = 0.707, s = 0.707
    Eigen::JacobiRotation<double> j_right;
    j_right.c() = 0.7071067811865475;
    j_right.s() = 0.7071067811865475;
    
    std::cout << "Right rotation: c = " << j_right.c() << ", s = " << j_right.s() << std::endl;
    
    // Apply right rotation to columns 0 and 1
    A.applyOnTheRight(0, 1, j_right);
    
    std::cout << "After right rotation:\n" << A << std::endl;
    std::cout << "A[0,0] = " << A(0,0) << ", A[1,1] = " << A(1,1) << std::endl;
    
    return 0;
}
