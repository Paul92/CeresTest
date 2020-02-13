
#include <string>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <ceres/ceres.h>

Eigen::MatrixXd readDense(const std::string &filename) {
    std::fstream file(filename);

    if (!file.is_open())
        throw std::invalid_argument("Could not open file " + filename);

    std::string matrix_type;
    file >> matrix_type;

    if (matrix_type != "DENSE")
        throw std::invalid_argument("Expected matrix type dense in " + filename);

    int rows, cols;
    file >> rows >> cols;

    if (rows < 0 || cols < 0)
        throw std::invalid_argument("Invalid matrix size in " + filename);

    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value;
            file >> value;
            mat(i, j) = value;
        }
    }

    return mat;
}

Eigen::MatrixXd load_data() {

    const Eigen::MatrixXd eq130 = readDense("test_data/eq130_rug.txt").col(0);
    const Eigen::MatrixXd eq121 = readDense("test_data/eq121_rug.txt").col(0);
    const Eigen::MatrixXd eq112 = readDense("test_data/eq112_rug.txt").col(0);
    const Eigen::MatrixXd eq103 = readDense("test_data/eq103_rug.txt").col(0);
    const Eigen::MatrixXd eq120 = readDense("test_data/eq120_rug.txt").col(0);
    const Eigen::MatrixXd eq111 = readDense("test_data/eq111_rug.txt").col(0);
    const Eigen::MatrixXd eq102 = readDense("test_data/eq102_rug.txt").col(0);
    const Eigen::MatrixXd eq110 = readDense("test_data/eq110_rug.txt").col(0);
    const Eigen::MatrixXd eq101 = readDense("test_data/eq101_rug.txt").col(0);
    const Eigen::MatrixXd eq100 = readDense("test_data/eq100_rug.txt").col(0);
    const Eigen::MatrixXd eq230 = readDense("test_data/eq230_rug.txt").col(0);
    const Eigen::MatrixXd eq221 = readDense("test_data/eq221_rug.txt").col(0);
    const Eigen::MatrixXd eq212 = readDense("test_data/eq212_rug.txt").col(0);
    const Eigen::MatrixXd eq203 = readDense("test_data/eq203_rug.txt").col(0);
    const Eigen::MatrixXd eq220 = readDense("test_data/eq220_rug.txt").col(0);
    const Eigen::MatrixXd eq211 = readDense("test_data/eq211_rug.txt").col(0);
    const Eigen::MatrixXd eq202 = readDense("test_data/eq202_rug.txt").col(0);
    const Eigen::MatrixXd eq210 = readDense("test_data/eq210_rug.txt").col(0);
    const Eigen::MatrixXd eq201 = readDense("test_data/eq201_rug.txt").col(0);
    const Eigen::MatrixXd eq200 = readDense("test_data/eq200_rug.txt").col(0);

    Eigen::MatrixXd pol_coeff(eq130.rows(), 20);

    pol_coeff.col(0)  = eq130;
    pol_coeff.col(1 ) = eq121;
    pol_coeff.col(2 ) = eq112;
    pol_coeff.col(3 ) = eq103;
    pol_coeff.col(4 ) = eq120;
    pol_coeff.col(5 ) = eq111;
    pol_coeff.col(6 ) = eq102;
    pol_coeff.col(7 ) = eq110;
    pol_coeff.col(8 ) = eq101;
    pol_coeff.col(9 ) = eq100;
    pol_coeff.col(10) = eq230;
    pol_coeff.col(11) = eq221;
    pol_coeff.col(12) = eq212;
    pol_coeff.col(13) = eq203;
    pol_coeff.col(14) = eq220;
    pol_coeff.col(15) = eq211;
    pol_coeff.col(16) = eq202;
    pol_coeff.col(17) = eq210;
    pol_coeff.col(18) = eq201;
    pol_coeff.col(19) = eq200;

    return pol_coeff;
}

template<typename T>
void evaluate_polynomial(const Eigen::MatrixXd &coeffs, T k1, T k2, T *residuals) {

    T k1_pow[] = {T(1), k1, k1*k1, k1*k1*k1};
    T k2_pow[] = {T(1), k2, k2*k2, k2*k2*k2};

    int k1_indices[] = {3, 2, 1, 0, 2, 1, 0, 1, 0, 0};
    int k2_indices[] = {0, 1, 2, 3, 0, 1, 2, 0, 1, 0};

    Eigen::Array<T, 10, 1> multiplier;
    for (int i = 0; i < 10; i++)
        multiplier(i) = k1_pow[k1_indices[i]] * k2_pow[k2_indices[i]];

    Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> map(residuals, coeffs.rows(), 2);

    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> left = coeffs.leftCols(10) * multiplier.matrix();
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> right = coeffs.rightCols(10) * multiplier.matrix();

    map << left, right;
}


class MyCostFunctor {
public:

    explicit MyCostFunctor(const Eigen::MatrixXd &pol_coeff) : pol_coeff(pol_coeff) { }

    template<typename T>
    bool operator()(const T* const k, T* residuals) const {
        evaluate_polynomial<T>(pol_coeff.topRows(10), k[0], k[1], residuals);
        return true;
    }

private:
    const Eigen::MatrixXd pol_coeff;
};


Eigen::ArrayXXd solve_polynomial(const Eigen::MatrixXd &coeff) {

    Eigen::ArrayXXd ks(2, 1);

    ks = 0;

    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<MyCostFunctor, 18, 2>(new MyCostFunctor(coeff));

    // Build the problem.
    ceres::Problem problem;
    problem.AddResidualBlock(cost_function, nullptr, ks.col(0).data());

    ceres::Solver::Options options;

    options.minimizer_progress_to_stdout = true;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 8;

    ceres::Solver::Summary summary;

    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    return ks;
}


int main() {

    const auto pol_coeffs = load_data();

    Eigen::ArrayXXd res = solve_polynomial(pol_coeffs).transpose();

    std::cout << res << std::endl;

    return 0;
}

