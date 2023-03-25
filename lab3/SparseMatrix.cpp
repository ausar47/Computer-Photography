#include <opencv2/opencv.hpp>
#include <vector>

class CRSMatrix {
public:
    // InitializeFromVector constructor
    CRSMatrix(const std::vector<int>& rows, const std::vector<int>& cols, const std::vector<double>& vals) {
        // check the validity of the input vectors
        assert(vals.size() == rows.size());
        assert(vals.size() == cols.size());

        std::vector<std::vector<double>> mat;
        this->nrows = *std::max_element(rows.begin(), rows.end()) + 1;  // let the number of rows equals the maximum value of rows + 1
        this->ncols = *std::max_element(cols.begin(), cols.end()) + 1;  // let the number of cols equals the maximum value of cols + 1
        
        // A structure to store a value and its indices
        struct Entry {
            double val; // value
            int row; // row index
            int col; // column index

            // A constructor to initialize an entry
            Entry(double v, int r, int c) {
                val = v;
                row = r;
                col = c;
            }

            // A comparison function to sort entries by row index and column index
            static bool compare(Entry& a, Entry& b) {
                if (a.row < b.row) return true; // smaller row index comes first
                if (a.row > b.row) return false; // larger row index comes last
                return a.col < b.col; // if row indices are equal, compare column indices
            }
        };
        
        std::vector<Entry> entries; // a vector to store entries
        for (int i = 0; i < vals.size(); ++i) { // loop through the vectors and create entries
            entries.push_back(Entry(vals[i], rows[i], cols[i]));
        }
        sort(entries.begin(), entries.end(), Entry::compare);
        int prev_row = -1; // a variable to store the previous row index
        for (Entry e : entries) { // loop through the sorted entries
            val.push_back(e.val); // copy the value to vals
            col_ind.push_back(e.col); // copy the column index to cols
            if (e.row != prev_row) { // if the row index is different from the previous one
                row_ptr.push_back(val.size() - 1); // store the current size of vals minus one as the row pointer
                prev_row = e.row; // update the previous row index
            }
        }
        row_ptr.push_back(vals.size()); // store the final size of vals as the last row pointer
    }

    // A constructor to initialize a CRS matrix from a dense matrix
    CRSMatrix(std::vector<std::vector<double>>& mat) {
        this->nrows = mat.size();
        this->ncols = mat[0].size();
        for (int i = 0; i < this->nrows; ++i) {
            this->row_ptr.push_back(this->val.size()); // store the current size of vals as the row pointer
            for (int j = 0; j < this->ncols; ++j) {
                if (mat[i][j] != 0) { // only store non-zero values
                    this->val.push_back(mat[i][j]);
                    this->col_ind.push_back(j);
                }
            }
        }
        this->row_ptr.push_back(val.size()); // store the final size of vals as the last row pointer
    }

    // A method to print the CRS matrix
    void print() {
        std::cout << "CRS Matrix Form: " << std::endl;
        std::cout << "val: ";
        for (double x : val) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "col_ind: ";
        for (int x : col_ind) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << "row_ptr: ";
        for (int x : row_ptr) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    // convert the CRS format to a dense matrix
    cv::Mat to_dense() {
        // get the shape of the matrix
        int m = nrows; // number of rows
        int n = ncols; // number of columns

        // create an empty matrix of shape (m, n)
        cv::Mat dense = cv::Mat::zeros(m, n, CV_64F);

        // loop through each row
        for (int i = 0; i < m; ++i) {
            // get the start and end index of the non-zero elements in vals
            int start = row_ptr[i];
            int end = row_ptr[i + 1];

            // get the corresponding column indices and values
            std::vector<int> cols(col_ind.begin() + start, col_ind.begin() + end);
            std::vector<double> vals(val.begin() + start, val.begin() + end);

            // assign the values to the dense matrix
            for (int j = 0; j < cols.size(); ++j) {
                dense.at<double>(i, cols[j]) = vals[j];
            }
        }

        return dense;
    }

    // access the element at (row, col) in the matrix
    double at(int row, int col) {
        // check the validity of the indices
        assert(row >= 0 && row < nrows);
        assert(col >= 0 && col < ncols);

        // get the start and end index of the non-zero elements in vals for row row
        int start = row_ptr[row];
        int end = row_ptr[row + 1];

        // get the corresponding column indices and values
        std::vector<int> cols(col_ind.begin() + start, col_ind.begin() + end);
        std::vector<double> vals(val.begin() + start, val.begin() + end);

        // find the index of col in cols
        auto it = std::find(cols.begin(), cols.end(), col);

        // if col is not found, return 0
        if (it == cols.end()) {
            return 0;
        }

        // otherwise, return the corresponding value in vals
        else {
            int k = it - cols.begin();
            return vals[k];
        }
    }

    // A method to insert a non-zero value into the CRS matrix
    void insert(double x, int row, int col) {
        // check the validity of the indices
        assert(row >= 0 && row < nrows);
        assert(col >= 0 && col < ncols);

        int start = row_ptr[row]; // the start index of the row row in vals and cols
        int end = row_ptr[row + 1]; // the end index of the row row in vals and cols
        for (int k = start; k < end; ++k) { // loop through the row row
            if (col_ind[k] == col) { // if the column index matches col
                if (x == 0) { // if the value to be inserted is zero, delete the existing value
                    val.erase(val.begin() + k);
                    col_ind.erase(col_ind.begin() + k);
                    for (int l = row + 1; l <= nrows; ++l) { // update the row pointers after row
                        --row_ptr[l];
                    }
                } else { // if the value to be inserted is not zero, replace the existing value
                    val[k] = x;
                }
                return; // exit the method
            }
            else if (col_ind[k] > col) { // if the column index is larger than col, insert the value before it
                if (x != 0) { // only insert non-zero values
                    val.insert(val.begin() + k, x);
                    col_ind.insert(col_ind.begin() + k, col);
                    for (int l = row + 1; l <= nrows; ++l) { // update the row pointers after row
                        ++row_ptr[l];
                    }
                }
                return; // exit the method
            }
        }
        // if the loop ends without finding or inserting the value, append it to the end of the row row
        if (x != 0) { // only insert non-zero values
            val.insert(val.begin() + end, x);
            col_ind.insert(col_ind.begin() + end, col);
            for (int l = row + 1; l <= nrows; ++l) { // update the row pointers after row
                ++row_ptr[l];
            }
        }
    }

    // A method to solve a linear system using Gauss-Seidel iteration
    std::vector<double> solve(const std::vector<double>& b, double delta, int max_iter) {
        // check if the vector b has the same size as the number of rows
        assert(b.size() == nrows);

        std::vector<double> x(nrows); // initialize a vector x to store the solution
        for (int k = 0; k < max_iter; ++k) { // loop for a maximum number of iterations
            for (int i = 0; i < nrows; ++i) { // loop through each row
                double sum = 0; // initialize a variable to store the sum of products
                int start = row_ptr[i]; // the start index of the row i in vals and cols
                int end = row_ptr[i + 1]; // the end index of the row i in vals and cols
                for (int j = start; j < end; ++j) { // loop through each non-zero element in the row i
                    if (col_ind[j] != i) { // if the column index is not equal to the row index
                        sum += val[j] * x[col_ind[j]]; // add the product of the value and the corresponding x element to the sum
                    }
                }
                // find the index of col in cols
                auto it = std::find(col_ind.begin() + start, col_ind.begin() + end, i);
                int k = it - col_ind.begin();
                double a_ii = val[k];
                x[i] = (b[i] - sum) / a_ii; // update the x element by subtracting the sum from b and dividing by the diagonal value
            }
            double err = 0; // initialize a variable to store the error
            for (int i = 0; i < nrows; ++i) { // loop through each row
                double res = b[i]; // initialize a variable to store the residual
                int start = row_ptr[i]; // the start index of the row i in vals and cols
                int end = row_ptr[i + 1]; // the end index of the row i in vals and cols
                for (int j = start; j < end; ++j) { // loop through each non-zero element in the row i
                    res -= val[j] * x[col_ind[j]]; // subtract the product of the value and the corresponding x element from the residual
                }
                err += res * res; // add the square of the residual to the error
            }
            err = sqrt(err); // take the square root of the error
            if (err <= delta) { // if the error is smaller than a given tolerance
                break; // exit the loop
            }
        }
        return x; // return the solution vector x
    }

    // A method to check the correctness of the solution
    bool check(const std::vector<double>& x, const std::vector<double>& b, double delta) {
        std::vector<double> r(b.size()); // initialize a vector r to store the residual
        for (int i = 0; i < nrows; ++i) { // loop through each row
            r[i] = b[i]; // copy the right-hand side value to r
            int start = row_ptr[i]; // the start index of the row i in vals and cols
            int end = row_ptr[i + 1]; // the end index of the row i in vals and cols
            for (int j = start; j < end; ++j) { // loop through each non-zero element in the row i
                r[i] -= val[j] * x[col_ind[j]]; // subtract the product of the value and the corresponding x element from r
            }
        }
        double norm = 0; // initialize a variable to store the norm of r
        for (double ri : r) { // loop through each element in r
            norm += ri * ri; // add the square of the element to norm
        }
        norm = sqrt(norm); // take the square root of norm
        std::cout << "norm: " << norm << std::endl;
        return norm <= delta; // return true if norm is smaller than delta, false otherwise
    }

    // CRS matrix times vector£¬result put in y
    void CRS_matvec(const std::vector<double>& x, std::vector<double>& y) {
        int n = nrows; // number of rows
        for (int i = 0; i < n; ++i) { // loop through each row
            y[i] = 0.0;
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) { // loop through each non-zero element in this row
                y[i] += val[k] * x[col_ind[k]]; // sum the product
            }
        }
    }

private:
    // attributes
    std::vector<double> val;    // non-zero values
    std::vector<int> col_ind;   // column indices
    std::vector<int> row_ptr;   // row ranges
    int nrows; // number of rows
    int ncols; // number of columns
};

// vector inner product
double dot(const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

// vector norm
double norm(const std::vector<double>& x) {
    return sqrt(dot(x, x));
}

// Conjugate gradient method solving Ax=b
// x is initialized as x0, store the result xk after iteration
std::vector<double> CG(CRSMatrix& A, const std::vector<double>& b, double delta, int max_iter) {
    int n = b.size(); // equation order
    std::vector<double> x(n);
    std::vector<double> r(n); // residual vector
    A.CRS_matvec(x, r); // compute Ax
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i]; // Compute b - Ax
    }
    double rho = dot(r, r);
    if (sqrt(rho) < delta) { // if r0 is sufficiently small, then return x0 as the result
        return x;
    }
    std::vector<double> p = r; // p0 = r0
    std::vector<double> q(n); // stores A * p_k

    for (int iter = 1; iter <= max_iter; ++iter) { // begin iteration
        A.CRS_matvec(p, q); // compute A * p_k
        double alpha = rho / dot(p, q); // compute step facto alpha_k
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i]; // x_k+1 = x_k + alpha_k * p_k
            r[i] -= alpha * q[i]; // r_k+1 = r_k - alpha_k * A * p_k
        }
        double rho_old = rho;
        rho = dot(r, r);

        if (sqrt(rho) <= delta) { // if r_k+1 is sufficiently small, then exit loop
            return x;
        }

        double beta = rho / rho_old;    // beta_k
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i]; // p_k+1 = r_k+1 + beta_k * p_k
        }
    }

    return x;
}

// A function to generate a random linear system of equations
// of the form Ax = b, where A is a n x n matrix and b is a n x 1 vector
// The function takes the size n as input and returns A and b as output
// The function guarantees that A is symmetric positive definite, so the system has a unique solution and Gauss-Seidel iteration converges
void generate_random_system(int n, double s, std::vector<std::vector<double>>& A, std::vector<double>& b) {
    // Initialize the random seed
    srand(time(NULL));

    // Resize A and b to the correct dimensions
    A.resize(n);
    for (int i = 0; i < n; ++i) {
        A[i].resize(n);
    }
    b.resize(n);

    // Generate a random lower triangular matrix L with positive diagonal entries
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                // Generate a random positive number in the range (0, 100]
                L[i][j] = (rand() % 100) + 1;
            } else {
                // Generate a random number in the range [-100, 100] with probability s
                double r = (double)rand() / RAND_MAX;
                if (r < s) {
                    L[i][j] = (rand() % 201) - 100;
                }
            }
        }
    }

    // Compute A as L L^T, which is symmetric positive definite
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                A[i][j] += L[i][k] * L[j][k];
            }
        }
    }

    // Generate a random solution x in the range [-100, 100]
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = (rand() % 201) - 100;
    }

    // Compute b as A x
    for (int i = 0; i < n; i++) {
        b[i] = 0;
        for (int j = 0; j < n; j++) {
            b[i] += A[i][j] * x[j];
        }
    }
}

// A function to print a matrix or a vector
template <typename T>
void print_matrix(const std::vector<std::vector<T>>& M) {
    std::cout << "Dense Matrix Form: " << std::endl;
    for (int i = 0; i < M.size(); ++i) {
        for (int j = 0; j < M[i].size(); ++j) {
            printf("%10.0f ", M[i][j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// A main function to test the CRS matrix class
int main(int argc, char* argv[]) {
    // CRS test cases
    //std::vector<int> rows = { 0, 0, 1, 1, 2, 2, 2, 3 };
    //std::vector<int> cols = { 0, 1, 1, 3, 2, 3, 4, 5 };
    //std::vector<double> vals = {10, 20, 30, 40, 50, 60, 70, 80};
    //CRSMatrix crs(rows, cols, vals); // create a CRS matrix from vectors
    //crs.print(); // print the CRS matrix
    // 
    // Gauss-Seidel test cases
    //std::vector<double> vals{ 4,-1,1,4,-8,1,-2,1,5 };
    //std::vector<int> cols{ 0,1,2,0,1,2,0,1,2 };
    //std::vector<int> rows{ 0,0,0,1,1,1,2,2,2 };
    //std::vector<double> b{ 7,-21,15 }; // the right-hand side vector
    //std::vector<double> vals{ 10,-1,2,-1,11,-1,3,2, -1, 10, -1, 3, -1, 8 };
    //std::vector<int> cols{ 0,1,2,0,1,2,3,0,1,2,3,1,2,3 };
    //std::vector<int> rows{ 0,0,0,1,1,1,1,2,2,2,2,3,3,3 };
    //std::vector<double> b{ 6, 25, -11, 15 }; // the right-hand side vector
    //CRSMatrix A(rows, cols, vals); // create a CRS matrix object from the vectors
    if (argc != 3) {
        std::cerr << "Input Format: matrix size, sparse factor" << std::endl;
        return -1;
    }
    std::vector<std::vector<double>> mat;
    std::vector<double>b;
    generate_random_system(atoi(argv[1]), atof(argv[2]), mat, b);
    print_matrix(mat);

    CRSMatrix A(mat);
    A.print();  // test the sparse matrix is successfully created
    double delta = 1e-6; // the tolerance
    int max_iter = 1000000; // the maximum number of iterations
    std::vector<double> x = A.solve(b, delta, max_iter); // call the solve method to get the solution vector x
    std::cout << std::endl << "Gaussian-Seidel method solution: " << "x = [";
    for (int i = 0; i < x.size(); ++i) { // loop through each element in x and print it
        if (i == x.size() - 1) {
            std::cout << x[i] << "]" << std::endl;
            break;
        }
        std::cout << x[i] << ", ";
    }
    std::string judge = A.check(x, b, delta) ? "Correct!" : "Incorrect!";   // check whether the solution is correct
    std::cout << "The solution is " << judge << std::endl;

    x.clear();
    x = CG(A, b, delta, max_iter);
    std::cout << std::endl << "Conjugate gradient method solution: " << "x = [";
    for (int i = 0; i < x.size(); ++i) { // loop through each element in x and print it
        if (i == x.size() - 1) {
            std::cout << x[i] << "]" << std::endl;
            break;
        }
        std::cout << x[i] << ", ";
    }
    judge = A.check(x, b, delta) ? "Correct!" : "Incorrect!";   // check whether the solution is correct
    std::cout << "The solution is " << judge << std::endl;
    return 0;
}