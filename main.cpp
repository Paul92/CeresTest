
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

std::array<Eigen::ArrayXXd, 20> load_data() {
    const auto eq130 = readDense("test_data/eq130_rug.txt");
    const auto eq121 = readDense("test_data/eq121_rug.txt");
    const auto eq112 = readDense("test_data/eq112_rug.txt");
    const auto eq103 = readDense("test_data/eq103_rug.txt");
    const auto eq120 = readDense("test_data/eq120_rug.txt");
    const auto eq111 = readDense("test_data/eq111_rug.txt");
    const auto eq102 = readDense("test_data/eq102_rug.txt");
    const auto eq110 = readDense("test_data/eq110_rug.txt");
    const auto eq101 = readDense("test_data/eq101_rug.txt");
    const auto eq100 = readDense("test_data/eq100_rug.txt");
    const auto eq230 = readDense("test_data/eq230_rug.txt");
    const auto eq221 = readDense("test_data/eq221_rug.txt");
    const auto eq212 = readDense("test_data/eq212_rug.txt");
    const auto eq203 = readDense("test_data/eq203_rug.txt");
    const auto eq220 = readDense("test_data/eq220_rug.txt");
    const auto eq211 = readDense("test_data/eq211_rug.txt");
    const auto eq202 = readDense("test_data/eq202_rug.txt");
    const auto eq210 = readDense("test_data/eq210_rug.txt");
    const auto eq201 = readDense("test_data/eq201_rug.txt");
    const auto eq200 = readDense("test_data/eq200_rug.txt");


    std::array<Eigen::ArrayXXd, 20> coeffs = {eq130, eq121, eq112, eq103, eq120, eq111, eq102, eq110, eq101, eq100,
                                              eq230, eq221, eq212, eq203, eq220, eq211, eq202, eq210, eq201, eq200};
    return coeffs;
}

template<typename T>
void evaluate_polynomial(const Eigen::MatrixXd &coeffs, T k1, T k2,
                         T *residuals) {

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

    explicit MyCostFunctor(const std::array<Eigen::ArrayXXd, 20> &coeff, int polynomial) : coeff(coeff), polynomial(polynomial) { }

    template<typename T>
    bool operator()(const T* const k, T* residuals) const {

        Eigen::MatrixXd pol_coeff(coeff[0].rows(), 20);

        for (std::size_t i = 0; i < coeff.size(); i++)
            pol_coeff.col(i) = coeff[i].col(polynomial);

        evaluate_polynomial<T>(pol_coeff.topRows(10), k[0], k[1], residuals);

        return true;
    }

private:
    const std::array<Eigen::ArrayXXd, 20> &coeff;
    const int polynomial;
};


Eigen::ArrayXXd solve_polynomial(const std::array<Eigen::ArrayXXd, 20> &coeff) {

    Eigen::ArrayXXd ks(2, coeff[0].cols());

    ks = 0;

    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<MyCostFunctor, 18, 2>(new MyCostFunctor(coeff, 0));

    // Build the problem.
    ceres::Problem problem;
    problem.AddResidualBlock(cost_function, nullptr, ks.col(0).data());

    ceres::Solver::Options options;

    options.minimizer_progress_to_stdout = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.num_threads = 8;

    ceres::Solver::Summary summary;

    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    return ks;
}


int main() {

    std::array<Eigen::ArrayXXd, 20> coeffs = load_data();

    Eigen::ArrayXXd res = solve_polynomial(coeffs).transpose();

    std::cout << res << std::endl;

    return 0;
}

