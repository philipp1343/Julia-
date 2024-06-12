using FileIO
using JLD2
using DataFrames

# Read result files and put them into a data frame
path = pwd()
files = readdir(path)
files = filter(f->occursin(".jld2",f),files)
dicts = map(f->load(joinpath(path,f)),files)
df = DataFrame(dicts)

display(df)

# TODO analyse the content of the data frame to produce figures, tables etc

