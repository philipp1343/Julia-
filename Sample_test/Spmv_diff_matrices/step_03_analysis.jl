using FileIO
using JLD2
using DataFrames
using CSV

# Read result files and put them into a data frame
path = pwd()
files = readdir(path)
files = filter(f -> occursin(".jld2", f), files)
dicts = map(f -> load(joinpath(path, f)), files)
df = DataFrame(dicts)

# Save the DataFrame into a .csv file
CSV.write("output.csv", df)

# Delete all .jld2 files
for file in files
    rm(joinpath(path, file))
end
