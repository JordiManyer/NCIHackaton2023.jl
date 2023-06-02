
using DrWatson
using DataFrames
using BSON
using Plots

basedir = "benchmarks"
df = collect_results(datadir(basedir))

line_styles = [:solid,:dash]
line_colour = [:red,:blue]

p11 = plot(ylabel="mul! time (s)",legend=true, title="D=2");
p12 = plot(legend=false, title="D=3");
p21 = plot(xlabel="FE order",ylabel="CG solver time (s)",legend=false);
p22 = plot(xlabel="FE order",legend=false);
plt = plot(p11,p12,p21,p22,layout=(2,2),legend=false);

dfgr = groupby(df,:D)
for (i,sub_df) in enumerate(dfgr)
  D = sub_df[1,:D]
  sort!(sub_df,:fe_order)
  for (j,var) in enumerate([:tmin_mul_sp,:tmin_mul_sf])
    plot!(plt[i],sub_df.fe_order,sub_df[!,var],lc=line_colour[j]);
  end
  for (j,var) in enumerate([:tmin_cg_sp,:tmin_cg_sf])
    plot!(plt[2+i],sub_df.fe_order,sub_df[!,var],lc=line_colour[j]);
  end
end

@show plt
