/**************************************************************
 * Compute eigenenergies of charged particle in magnetic field
 * Use MPI, run on multiple nodes
 *
 * Jing Cui
 **************************************************************/
//mpic++ -std=c++11 SingleProcess.cpp ChargedParticle.cpp -lgsl -lgslcblas -static-libstdc++

#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cmath>
#include<chrono>
#include <mpi.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include "SingleProcess.h"

using namespace std;

const double eps = 0.0001;
const int nrestart = 10000;
const int kmax = 4;
const int pmax = 20;
const int kpmax = kmax + pmax;
const double L = 10;
const int Ntot = 256;
const double Bfield = 0.00002;


/* Convert vector index to 2D Cartesian coordinates (x,y) */
vector<int> index_to_cartesian(const int& i, const int& N);
/* Compute H matrix */
void lanczos(int start, int end, int rank, int M, int N, gsl_matrix *H, gsl_matrix *V, gsl_matrix *V_local, int np) ;
/* Matrix Multiplication */
void gsl_matrix_mul(const gsl_matrix *a, const gsl_matrix *b, gsl_matrix *c);



int main(int argc, char *argv[]) {

  auto start = std::chrono::high_resolution_clock::now();

    // Set up the grid
    // Physical length of the system is (2*L) * (2*L)
    // Node number is M * M = np, assume N is even
    // Grid Size in each node is N * N
    // Ntot = M * N
    
    // Initialize MPI:
    MPI::Init(argc, argv);
    int rank = MPI::COMM_WORLD.Get_rank();
    int np = MPI::COMM_WORLD.Get_size();
    int M = sqrt(np);
    int N = Ntot / M;
    bool converged = false;
    
    gsl_matrix *H;
    gsl_matrix *QR;
    gsl_matrix *Q;
    gsl_matrix *Qt;
    gsl_matrix *R;
    gsl_matrix *Hplus;
    gsl_matrix *I;
    gsl_vector *tau;
    gsl_matrix *Q_new;
    gsl_matrix *Hplus_new;
    gsl_vector *eval;
    gsl_matrix *evec;
    gsl_eigen_symmv_workspace *w;
    gsl_matrix *V;
    gsl_matrix *V_local;
    gsl_matrix *V_local_new;

    V_local = gsl_matrix_alloc(N * N * 2, kpmax+1);
    V_local_new = gsl_matrix_alloc(N * N * 2, kpmax+1);
    H = gsl_matrix_alloc(kpmax, kpmax);
    Q = gsl_matrix_alloc(kpmax, kpmax);
    
    
    if (rank == 0) {
        
        V = gsl_matrix_alloc(Ntot * Ntot * 2, kpmax+1);
        QR = gsl_matrix_alloc (kpmax, kpmax);
        Qt = gsl_matrix_alloc (kpmax, kpmax);
        R = gsl_matrix_alloc (kpmax, kpmax);
        Hplus = gsl_matrix_alloc (kpmax, kpmax);
        I = gsl_matrix_alloc (kpmax, kpmax);
        tau = gsl_vector_alloc (kpmax);
        Q_new = gsl_matrix_alloc (kpmax, kpmax);
        Hplus_new = gsl_matrix_alloc (kpmax, kpmax);
        
        eval = gsl_vector_alloc (kpmax);
        evec = gsl_matrix_alloc (kpmax, kpmax);
        w = gsl_eigen_symmv_alloc (kpmax);
    }
    
    
    vector<double> EigenValue(kmax);
    vector<vector<double>> EigenVector(kmax, vector<double>(2 * Ntot * Ntot));
    int eknown;
    
    // Initial Lanczos step:
    lanczos(0, kpmax, rank, M, N, H, V, V_local, np);
    
    for (int itr = 0; itr < nrestart; itr++) {
        if (rank == 0) {
            gsl_eigen_symmv (H, eval, evec, w);
            
            gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);
            
            for (int i = 0; i < kmax; i++) {
                double eval_i = gsl_vector_get (eval, i);
                cout << "Eigenvalue " << i << " = " << eval_i << endl;
            }
            
            /****************    Check for convergence of eigenvalues       *************************/
            /*************    If converged, save eigenvector and eigenvalue     *********************/
            eknown = 0;
            double tmp;
            
            if (itr > 0) {
                for (int k = 0; k < kmax; k++) {
                    tmp = std::abs ((EigenValue[k] - gsl_vector_get (eval, k))/EigenValue[k]);
                    if ( tmp < eps )
                        eknown = eknown + 1;
                    EigenValue[k] = gsl_vector_get (eval, k);
                }
            }
            cout << eknown << " Eigenvalues Converged" << endl;
            
            if ( eknown == kmax ) {
                for (int k = 0; k < kmax; k++) {
                    EigenValue[k] = gsl_vector_get (eval, k);
                    
                    for (int n = 0; n < Ntot * Ntot * 2; n++) {
                        EigenVector[k][n] = 0.0;
                        for (int m = 0; m < kpmax; m++) {
                            EigenVector[k][n] += gsl_matrix_get(V, n, m) * gsl_matrix_get ( evec, m, k);
                        }
                    }
                    
                }
                
                converged = true;
                
            }
            /*****************************************************************************************/
            
            gsl_matrix_memcpy (Hplus, H);
            gsl_matrix_set_identity (Q);
            
            for (int p = kmax; p < kpmax; p++) {
                
                /* Do pmax shifts */
                gsl_matrix_set_identity (I);
                gsl_matrix_memcpy (QR, Hplus);
                gsl_matrix_scale (I, gsl_vector_get (eval, p));
                gsl_matrix_sub (QR, I);
                gsl_linalg_QR_decomp (QR, tau);
                gsl_linalg_QR_unpack (QR, tau, Qt, R);
                
                gsl_matrix_mul (Q, Qt, Q_new);
                gsl_matrix_memcpy (Q, Q_new);
                gsl_matrix_mul (Hplus, Qt, Hplus_new);
                gsl_matrix_memcpy (Hplus, Hplus_new);
                gsl_matrix_transpose (Qt);
                gsl_matrix_mul (Qt, Hplus, Hplus_new);
                gsl_matrix_memcpy (Hplus, Hplus_new);
            }
            cout << "mark1" << endl;
            cout << "mark2" << endl;
              for(int j = 0; j < kmax; j++) {
                gsl_matrix_set(H, j, j, gsl_matrix_get (Hplus, j, j));
                gsl_matrix_set(H, j+1, j, gsl_matrix_get (Hplus, j+1, j));
                gsl_matrix_set(H, j, j+1, gsl_matrix_get (Hplus, j, j+1));
            }
            cout << "mark3" << endl;
            cout << "Lanczos Iteration : " << itr << endl;
        }
        
        MPI::COMM_WORLD.Bcast(&converged, 1, MPI::BOOL, 0);
        if(converged) break;
        else {
            MPI::COMM_WORLD.Bcast(H->data, kpmax * kpmax, MPI::DOUBLE, 0);
            MPI::COMM_WORLD.Bcast(Q->data, kpmax * kpmax, MPI::DOUBLE, 0);
            gsl_matrix_mul(V_local, Q, V_local_new);
            gsl_matrix_memcpy(V_local, V_local_new);
            lanczos(kmax-1, kpmax, rank, M, N, H, V, V_local, np);
        }
    }
    
    if (rank == 0) {
        /* Output groundstate to file */
        ostringstream convert;
        convert << Ntot;
        string index = convert.str();
        ofstream file[kmax];
        string filename[kmax];
        int label1;
        int label2;
        int rk;
        int eIndex;
        for(int k = 0; k < kmax; k++) {
            convert.str("");
            convert.clear();
            convert << k;
            filename[k] = "eigenvector" + convert.str() + "_" +  index + ".txt";
            file[k].open(filename[k]);
            for(int i = 0; i < Ntot; i++) {
                for(int j = 0; j < Ntot; j++) {
                    label1 = i / N;
                    label2 = j / N;
                    rk = label1 * M + label2;
                    eIndex = rk * N * N + (i % N) * N + j % N;
                    file[k]<<EigenVector[k][2*eIndex] << " " <<EigenVector[k][2*eIndex+1] << " ";
                }
                file[k]<<endl;
            }
            file[k].close();
        }
        
    }
    gsl_matrix_free (H);
    gsl_matrix_free (V_local);
    gsl_matrix_free (Q);
    gsl_matrix_free (V_local_new);
    if(rank == 0) {
        
        gsl_eigen_symmv_free (w);
        gsl_vector_free (eval);
        gsl_matrix_free (evec);
        gsl_vector_free (tau);
        gsl_matrix_free (QR);
        gsl_matrix_free (R);
        gsl_matrix_free (Hplus);
        gsl_matrix_free (Qt);
        gsl_matrix_free (Hplus_new);
        gsl_matrix_free (Q_new);
        gsl_matrix_free (I);
        gsl_matrix_free (V);
    }
    
  
    
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0)
      cout << "Time (microseconds): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << endl;
   
    // Close MPI
    MPI::Finalize();
}






/* Convert vector index to 2D Cartesian coordinates (x,y) */
vector<int> index_to_cartesian(const int& i, const int& N)
{
    std::vector<int> v(3);
    int x = i / N;
    int y = i % N;
    v[0] = x;
    v[1] = y;
    return v;
}

/* Matrix Multiplication */
void gsl_matrix_mul(const gsl_matrix *a, const gsl_matrix *b, gsl_matrix *c)
{
    for (size_t i = 0; i < a->size1; i++)
    {
        for (size_t j = 0; j < b->size2; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < b->size1; k++)
            {
                sum += gsl_matrix_get (a, i, k) * gsl_matrix_get (b, k, j);
            }
            gsl_matrix_set (c, i, j, sum);
        }
    }
}


/* Compute H matrix, collective psi into V (when rank==0) */
void lanczos(int start, int end, int rank, int M, int N, gsl_matrix *H, gsl_matrix *V, gsl_matrix *V_local, int np) {
    int top = rank - M;
    int bottom = rank + M;
    int left = rank - 1;
    int right = rank + 1;
    int N2 = N * N * 2;

    double a = 0.0;
    double b = 0.0;
    double a_tot = 0.0;
    double b_tot = 0.0;
    
    vector<vector<double>> Global_Psi(kpmax+1, vector<double>(Ntot * Ntot * 2));
    vector<vector<double>> Local_Psi(kpmax+1, vector<double>(N2));
    
    vector<double> topSendBuffer(N*2, 0.0);
    vector<double> bottomSendBuffer(N*2, 0.0);
    vector<double> leftSendBuffer(N*2, 0.0);
    vector<double> rightSendBuffer(N*2, 0.0);
    
    vector<double> topReceiveBuffer(N*2, 0.0);
    vector<double> bottomReceiveBuffer(N*2, 0.0);
    vector<double> leftReceiveBuffer(N*2, 0.0);
    vector<double> rightReceiveBuffer(N*2, 0.0);
    
    SingleProcess sp (N, rank, M, L, kmax, pmax, Bfield, np);
    
    if (start > 0) {
    
        for (int itr = 0; itr <= start; itr++) {
            for (int j = 0; j < N2; j++) {
                Local_Psi[itr][j] = gsl_matrix_get(V_local, j, itr);
            }
            sp.setAlpha(gsl_matrix_get(H, itr, itr));
            sp.setBeta(gsl_matrix_get(H, itr, itr+1));
            sp.setPsi(Local_Psi[itr]);
        }
    }
    
    if(start > 0) {sp.counts -= 1;}
    
    for (int itr = start; itr < end; itr++ ) {
        
        a_tot = 0.0;
        b_tot = 0.0;
        
        topSendBuffer = sp.getTopBoundary();
        bottomSendBuffer = sp.getBottomBoundary();
        leftSendBuffer = sp.getLeftBoundary();
        rightSendBuffer = sp.getRightBoundary();
        
        
        // Use MPI blocking send and receive
        // Send to top:
        if (rank / M % 2 == 1) { if(rank / M != 0) MPI::COMM_WORLD.Send(&topSendBuffer.front(), N*2, MPI::DOUBLE, top, 0); }
        else { if(rank / M != M-1) MPI::COMM_WORLD.Recv(&bottomReceiveBuffer.front(), N*2, MPI::DOUBLE, bottom, 0); }
        if (rank / M % 2 == 0) { if(rank / M != 0) MPI::COMM_WORLD.Send(&topSendBuffer.front(), N*2, MPI::DOUBLE, top, 0); }
        else { if(rank / M != M-1) MPI::COMM_WORLD.Recv(&bottomReceiveBuffer.front(), N*2, MPI::DOUBLE, bottom, 0); }

        // Send to bottom:
        if (rank / M % 2 == 0) { if(rank / M != M-1) MPI::COMM_WORLD.Send(&bottomSendBuffer.front(), N*2, MPI::DOUBLE, bottom, 0); }
        else { if(rank / M != 0) MPI::COMM_WORLD.Recv(&topReceiveBuffer.front(), N*2, MPI::DOUBLE, top, 0); }
        if (rank / M % 2 == 1) { if(rank / M != M-1) MPI::COMM_WORLD.Send(&bottomSendBuffer.front(), N*2, MPI::DOUBLE, bottom, 0); }
        else { if(rank / M != 0) MPI::COMM_WORLD.Recv(&topReceiveBuffer.front(), N*2, MPI::DOUBLE, top, 0); }
        
        
        //Send to right:
        if (rank % M % 2 == 0) { if(rank % M != M-1) MPI::COMM_WORLD.Send(&rightSendBuffer.front(), N*2, MPI::DOUBLE, right, 2); }
        else { if(rank % M != 0) MPI::COMM_WORLD.Recv(&leftReceiveBuffer.front(), N*2, MPI::DOUBLE, left, 2); }
        if (rank % M % 2 == 1) { if(rank % M != M-1) MPI::COMM_WORLD.Send(&rightSendBuffer.front(), N*2, MPI::DOUBLE, right, 2); }
        else { if(rank % M != 0) MPI::COMM_WORLD.Recv(&leftReceiveBuffer.front(), N*2, MPI::DOUBLE, left, 2); }

        //Send to left:
        if (rank % M % 2 == 1) { if(rank % M != 0) MPI::COMM_WORLD.Send(&leftSendBuffer.front(), N*2, MPI::DOUBLE, left, 0); }
        else { if(rank % M != M-1) MPI::COMM_WORLD.Recv(&rightReceiveBuffer.front(), N*2, MPI::DOUBLE, right, 0); }
        if (rank % M % 2 == 0) { if(rank % M != 0) MPI::COMM_WORLD.Send(&leftSendBuffer.front(), N*2, MPI::DOUBLE, left, 0); }
        else { if(rank % M != M-1) MPI::COMM_WORLD.Recv(&rightReceiveBuffer.front(), N*2, MPI::DOUBLE, right, 0); }

        
        sp.setTopBoundary(topReceiveBuffer);
        sp.setBottomBoundary(bottomReceiveBuffer);
        sp.setLeftBoundary(leftReceiveBuffer);
        sp.setRightBoundary(rightReceiveBuffer);
        
        
        sp.applyHamiltonian();
        a = sp.getAlpha();
        //cout << "local a from rank :" << rank << " a = " << a << endl;
        
        // Master receives a from each worker and computes sum
        MPI::COMM_WORLD.Reduce(&a, &a_tot, 1, MPI::DOUBLE, MPI::SUM, 0);
        MPI::COMM_WORLD.Bcast(&a_tot, 1, MPI::DOUBLE, 0);
        
        if(rank == 0) {
            gsl_matrix_set(H, itr, itr, a_tot);
        }
        sp.setAlpha(a_tot);
        //cout << "global a from rank :" << rank << " a_tot = " << a_tot << endl;
        sp.updatePsiA();
        
        b = sp.getBeta();
        //cout << "local b from rank :" << rank << " b = " << b << endl;

        MPI::COMM_WORLD.Reduce(&b, &b_tot, 1, MPI::DOUBLE, MPI::SUM, 0);
        if (rank == 0) b_tot = sqrt(b_tot);
        MPI::COMM_WORLD.Bcast(&b_tot, 1, MPI::DOUBLE, 0);
        if(rank == 0 && itr < kpmax-1) {
            gsl_matrix_set(H, itr, itr+1, b_tot);
            gsl_matrix_set(H, itr+1, itr, b_tot);
        }
        sp.setBeta(b_tot);
        //cout << "global b from rank :" << rank << " b_tot = " << b_tot << endl;
        
        vector<double> temp(kpmax);
        MPI::COMM_WORLD.Reduce(&(sp.orthogonalFactor.front()), &temp.front(), kpmax, MPI::DOUBLE, MPI::SUM, 0);
        sp.orthogonalFactor = temp;
        MPI::COMM_WORLD.Bcast(&sp.orthogonalFactor.front(), kpmax, MPI::DOUBLE, 0);
        sp.updatePsiB();
    }
    
    for (int i = 0; i <= end; i++) {
        MPI::COMM_WORLD.Gather(&(sp.getPsi(i).front()), N2, MPI::DOUBLE, &(Global_Psi[i].front()), N2, MPI::DOUBLE, 0);
        if( rank == 0) {
            for (int j = 0; j < Ntot * Ntot * 2; j++) {
                gsl_matrix_set(V, j, i, Global_Psi[i][j]);
            }
        }
	Local_Psi[i] = sp.getPsi(i);
	for (int j = 0; j < N * N * 2; j++) {
	  gsl_matrix_set(V_local, j, i, Local_Psi[i][j]);
	}
    }
}




