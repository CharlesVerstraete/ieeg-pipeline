# julia 1.10
# -*- coding: utf-8 -*-

"""
Description:
    import necessary toolboxes

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05

"""

using Turing
using StatsPlots
using DataFrames
using CSV
using FileIO
using JLD2
using Plots
using StatsBase
using LinearAlgebra
using StatsFuns
using SpecialFunctions
using ColorSchemes
using HypothesisTests
using Serialization
using Printf
using ForwardDiff
using Measures


## Load config
include("config.jl");

## Load utils
include("utils/model_helper.jl");
include("utils/data_helper.jl");
include("utils/simulation_helper.jl");

## Load models
include("model/stratinf_selection_greedy.jl")
include("model/stratinf_transition.jl");
include("model/stratinf_selection.jl");
include("model/stratinf_entropy.jl");
include("model/hmm.jl");
