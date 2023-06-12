
using Plots

fe_order = [4,5,6,7]
t_gpu    = [0.003,0.0051,0.00805,0.014]
t_cpu    = [0.5,0.74,0.91,1.09]
t_sparse = [0.79,1.64,3.09,Inf]

plt = plot(xlabel="FE order", ylabel="Speedup");
plot!(fe_order,t_cpu ./ t_gpu,labels="cpu sumfac vs gpu");
plot!(fe_order,t_sparse ./ t_gpu,labels="cpu sparse vs gpu");

@show plt

savefig(plt,"figures/final_speedup.png")
