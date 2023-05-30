
using FileIO, DrWatson
using ProfileView, FlameGraphs, PProf

filenames = ["profile_sparse.jlprof","profile_sumfac_cpu.jlprof"]
profiles  = map(f->ProfileView.load(datadir(f)),filenames)
graphs    = map(p->flamegraph(first(p);lidict=last(p)),profiles)

# CG + sparse matrix
ProfileView.view(graphs[1])
pprof(graphs[1];out=datadir("profile_sparse.pb.gz"),ui_relative_percentages =true)

# CG + sum factorization
ProfileView.view(graphs[2])
pprof(graphs[2];out=datadir("profile_sumfac_cpu.pb.gz"),ui_relative_percentages =true)
