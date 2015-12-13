/***************************************************
 * SingleProcess.h
 *
 *
 * Cui Jing
 **************************************************/


#ifndef SingleProcess_h
#define SingleProcess_h

#include <vector>
#include <gsl/gsl_matrix.h>
#include <iostream>

using std::vector;

class SingleProcess {
public:
    
    int counts;
    
    /* Constructor and Destructor */
    SingleProcess(int N, double Ymin, double Ymax, int kmax, int pmax, double Bfield);
    ~SingleProcess() {};
    
    /* Apply Hamiltonian to psi[counts] and get new psi[counts+1] */
    void applyHamiltonian();
    
    /* Update psi[counts+1] according to alpha and beta, increment counts */
    void updatePsiA();
    void updatePsiB();

    /* Inner product of psi[i] and psi[j], only take real part. conjugate(transpose(psi[i])) * psi[j]
     * Because those inner product that matters in the computation all return real numbers.
     */
    double innerProduct(int i, int j);

    /* Set alpha and beta */
    void setAlpha(double a) { alpha[counts] = a; };
    void setBeta(double b) { beta[counts] = b; };
    
    /* Get alpha and beta */
    double getAlpha() {return alpha[counts]; }
    double getBeta() {return beta[counts]; }
    
    /* Set psi, increment counts */
    void setPsi(const vector<double>& Psi);
    
    /* Get psi by index */
    std::vector<double> getPsi(int i) {return psi[i]; }
    
    /* Set boundary vectors */
    void setTopBoundary(const vector<double>& tb) { topBoundary = tb; };
    void setBottomBoundary(const vector<double>& bb) { bottomBoundary = bb; };
    void setLeftBoundary(const vector<double>& lb) { leftBoundary = lb; };
    void setRightBoundary(const vector<double>& rb) { rightBoundary = rb; };
    
    /* Get boundary vectors */
    std::vector<double> getTopBoundary();
    std::vector<double> getBottomBoundary();
    std::vector<double> getLeftBoundary();
    std::vector<double> getRightBoundary();
    
    
    
private:
    int N, kmax, pmax, kpmax;
    double Ymin, Ymax;
    double B;
    std::vector<double> alpha, beta;

    /* Complex numbers, include real part and imaginary part */
    std::vector<std::vector<double>> psi;
    std::vector<double> topBoundary;
    std::vector<double> bottomBoundary;
    std::vector<double> leftBoundary;
    std::vector<double> rightBoundary;
    std::vector<double> Y;
    
};


extern const double h;
extern const double m;
extern const double q;


#endif /* SingleProcess_h */
