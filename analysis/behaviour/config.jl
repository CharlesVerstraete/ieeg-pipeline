# julia 1.10
# -*- coding: utf-8 -*-

"""
config.jl
================

Description:
    create environment variable 

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05
"""


##################################################
# Path definition
# DATA_PATH = "/home/cverstraete/nasShare/projects/cverstraete/data";
DATA_PATH = "/Users/charles.verstraete/Documents/w3_iEEG/data_cluster"
# CHAIN_PATH = "/home/cverstraete/nasShare/projects/cverstraete/behavior/outputs/chains";
# FIGURE_PATH = "/home/cverstraete/nasShare/projects/cverstraete/behavior/outputs/figures";
# TABLE_PATH = "/home/cverstraete/nasShare/projects/cverstraete/behavior/outputs/tables";

##################################################
# Model variables
MODEL_SPACE = [:stratinf_transition, :stratinf_selection]


# Hidden states of the HMM (corresponding with strategies) 
const HIDDEN_STATES = vec(reshape([[i, j, k] for i = 1:3 for j = 1:3 for k = 1:3], 27, 1))

# All possible sets of strategies
const ALL_SETS = [begin
    m = zeros(Int, 3, 3)
    m[v[1], 1] = 1
    m[v[2], 2] = 1 
    m[v[3], 3] = 1
    m
end for v in HIDDEN_STATES]

# Build the transition matrix between strategies 
const NSETS = length(ALL_SETS)
tmp = fill(4, NSETS+1, NSETS+1)
for i = 1:NSETS, j = 1:NSETS
    tmp[i,j] = sum(HIDDEN_STATES[i] .≠ HIDDEN_STATES[j])
end

const HTRANS = tmp
const HIDX = [findall(HTRANS .== i) for i = 0:4]
const HN = [sum(HTRANS[:,1] .== i) for i = 0:4]

const pmin = 1e-40

# Switch types between strategies
const TYPE_IDX = hcat(
    [findall([sum(any(ALL_SETS[i] .≠ ALL_SETS[j], dims=1)) == 1 for j = 1:27]) for i = 1:27],
    [findall([sum(any(ALL_SETS[i] .≠ ALL_SETS[j], dims=1)) == 2 for j = 1:27]) for i = 1:27],
    [findall([sum(any(ALL_SETS[i] .≠ ALL_SETS[j], dims=1)) == 3 for j = 1:27]) for i = 1:27]
)

const SWITCH_TYPES = Dict(
    1 => "overlap",
    2 => "overlap", 
    3 => "global",
    4 => "random"
)

const RANDOM_STATE = 28
const MIN_OBSERVATIONS = 5
