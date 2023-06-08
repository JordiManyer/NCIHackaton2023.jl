include("drivers.jl")

using FileIO, DrWatson
using Profile, ProfileView

n = 20
D = 3                             # Problem dimension
fe_orders   = Tuple(fill(4,D))    # FE element orders
quad_orders = Tuple(fill(6,D))    # Quadrature orders 

# A) CG using sparse matrix
A,b = setup_sparse(n,D,fe_orders,quad_orders)
compute_solution(A,b)
Profile.clear(); Profile.@profile compute_solution(A,b);
save(datadir("profile_sparse.jlprof"), Profile.retrieve()...)
# Profile.clear(); ProfileView.@profview compute_solution(A,b) 

# B) CG using sum-factorization based matrix
A_lazy,b = setup_sumfac_cpu(n,D,fe_orders,quad_orders)
compute_solution(A_lazy,b)
Profile.clear(); Profile.@profile compute_solution(A_lazy,b);
save(datadir("profile_sumfac_cpu.jlprof"), Profile.retrieve()...)
# Profile.clear(); ProfileView.@profview compute_solution(A_lazy,b)
