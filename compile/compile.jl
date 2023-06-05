using PackageCompiler
create_sysimage(:NCIHackaton2023,
  sysimage_path=joinpath(@__DIR__,"..","Environment.so"),
  precompile_execution_file=joinpath(@__DIR__,"precompile_script.jl"))
