#=
Time-series aware cross-fitting strategies for Dynamic DML.

Implements cross-validation splits that respect temporal ordering to prevent
information leakage from future to past observations.

Reference:
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.
=#

module CrossFitting

export BlockedTimeSeriesSplit, RollingOriginSplit, PanelStratifiedSplit
export ProgressiveBlockSplit
export split_indices, n_splits, get_cross_validator

"""
    BlockedTimeSeriesSplit

Time-blocked cross-validation for single time series.
Divides into K contiguous blocks.

# Fields
- `n_splits::Int`: Number of folds/blocks
- `gap::Int`: Gap between train and test sets
"""
struct BlockedTimeSeriesSplit
    n_splits::Int
    gap::Int

    function BlockedTimeSeriesSplit(; n_splits::Int=5, gap::Int=0)
        n_splits < 2 && error("n_splits must be at least 2, got $n_splits")
        gap < 0 && error("gap must be non-negative, got $gap")
        new(n_splits, gap)
    end
end

"""
    split_indices(cv::BlockedTimeSeriesSplit, n_samples::Int)

Generate train/test indices for blocked time series split.
"""
function split_indices(cv::BlockedTimeSeriesSplit, n_samples::Int)
    block_size = n_samples ÷ cv.n_splits
    indices = collect(1:n_samples)

    splits = Tuple{Vector{Int}, Vector{Int}}[]

    for i in 1:cv.n_splits
        test_start = (i - 1) * block_size + 1
        test_end = i == cv.n_splits ? n_samples : i * block_size

        test_idx = indices[test_start:test_end]

        # Training: all other blocks with gap
        train_before_end = max(1, test_start - cv.gap)
        train_after_start = min(n_samples + 1, test_end + cv.gap + 1)

        train_idx = vcat(
            indices[1:train_before_end-1],
            indices[train_after_start:end]
        )

        if !isempty(train_idx)
            push!(splits, (train_idx, test_idx))
        end
    end

    return splits
end

n_splits(cv::BlockedTimeSeriesSplit) = cv.n_splits


"""
    RollingOriginSplit

Expanding window cross-validation (walk-forward validation).
Training window expands forward, always testing on future data.

# Fields
- `initial_window::Int`: Minimum training set size
- `step::Int`: Observations added each iteration
- `horizon::Int`: Test set size
- `gap::Int`: Gap between train and test
- `max_train_size::Union{Int,Nothing}`: Maximum training size
"""
struct RollingOriginSplit
    initial_window::Int
    step::Int
    horizon::Int
    gap::Int
    max_train_size::Union{Int,Nothing}

    function RollingOriginSplit(;
        initial_window::Int,
        step::Int=1,
        horizon::Int=1,
        gap::Int=0,
        max_train_size::Union{Int,Nothing}=nothing
    )
        initial_window < 1 && error("initial_window must be positive, got $initial_window")
        step < 1 && error("step must be positive, got $step")
        horizon < 1 && error("horizon must be positive, got $horizon")
        gap < 0 && error("gap must be non-negative, got $gap")
        new(initial_window, step, horizon, gap, max_train_size)
    end
end

function split_indices(cv::RollingOriginSplit, n_samples::Int)
    indices = collect(1:n_samples)
    splits = Tuple{Vector{Int}, Vector{Int}}[]

    train_end = cv.initial_window

    while train_end + cv.gap + cv.horizon <= n_samples
        train_start = 1
        if !isnothing(cv.max_train_size)
            train_start = max(1, train_end - cv.max_train_size + 1)
        end
        train_idx = indices[train_start:train_end]

        test_start = train_end + cv.gap + 1
        test_end = min(test_start + cv.horizon - 1, n_samples)
        test_idx = indices[test_start:test_end]

        push!(splits, (train_idx, test_idx))
        train_end += cv.step
    end

    return splits
end

function n_splits(cv::RollingOriginSplit, n_samples::Int)
    count = 0
    train_end = cv.initial_window
    while train_end + cv.gap + cv.horizon <= n_samples
        count += 1
        train_end += cv.step
    end
    return count
end


"""
    PanelStratifiedSplit

Cross-fitting for panel data by stratifying on units.
Splits by unit rather than time, preserving temporal structure within units.

# Fields
- `n_splits::Int`: Number of folds
- `shuffle::Bool`: Whether to shuffle units
- `seed::Union{Int,Nothing}`: Random seed
"""
struct PanelStratifiedSplit
    n_splits::Int
    shuffle::Bool
    seed::Union{Int,Nothing}

    function PanelStratifiedSplit(; n_splits::Int=5, shuffle::Bool=false, seed::Union{Int,Nothing}=nothing)
        n_splits < 2 && error("n_splits must be at least 2, got $n_splits")
        new(n_splits, shuffle, seed)
    end
end

function split_indices(cv::PanelStratifiedSplit, n_samples::Int, unit_id::Vector{Int})
    unique_units = unique(unit_id)
    n_units = length(unique_units)

    n_units < cv.n_splits && error("Number of units ($n_units) must be at least n_splits ($(cv.n_splits))")

    # Optionally shuffle units
    if cv.shuffle
        if !isnothing(cv.seed)
            Random.seed!(cv.seed)
        end
        unit_order = shuffle(unique_units)
    else
        unit_order = unique_units
    end

    fold_size = n_units ÷ cv.n_splits
    splits = Tuple{Vector{Int}, Vector{Int}}[]

    for i in 1:cv.n_splits
        test_start = (i - 1) * fold_size + 1
        test_end = i == cv.n_splits ? n_units : i * fold_size
        test_units = unit_order[test_start:test_end]

        train_units = setdiff(unique_units, test_units)

        test_mask = [u in test_units for u in unit_id]
        train_mask = [u in train_units for u in unit_id]

        test_idx = findall(test_mask)
        train_idx = findall(train_mask)

        push!(splits, (train_idx, test_idx))
    end

    return splits
end

n_splits(cv::PanelStratifiedSplit) = cv.n_splits


"""
    ProgressiveBlockSplit

Progressive block cross-fitting for single long time series.
For block b, trains on blocks 1..b-1, predicts for block b.

# Fields
- `n_blocks::Int`: Number of blocks
- `min_train_blocks::Int`: Minimum training blocks before starting predictions
"""
struct ProgressiveBlockSplit
    n_blocks::Int
    min_train_blocks::Int

    function ProgressiveBlockSplit(; n_blocks::Int=10, min_train_blocks::Int=2)
        n_blocks < 3 && error("n_blocks must be at least 3, got $n_blocks")
        min_train_blocks < 1 && error("min_train_blocks must be positive, got $min_train_blocks")
        min_train_blocks >= n_blocks && error("min_train_blocks ($min_train_blocks) must be less than n_blocks ($n_blocks)")
        new(n_blocks, min_train_blocks)
    end
end

function split_indices(cv::ProgressiveBlockSplit, n_samples::Int)
    block_size = n_samples ÷ cv.n_blocks
    indices = collect(1:n_samples)
    splits = Tuple{Vector{Int}, Vector{Int}}[]

    for b in cv.min_train_blocks:(cv.n_blocks - 1)
        train_end = b * block_size
        train_idx = indices[1:train_end]

        test_start = b * block_size + 1
        test_end = b == cv.n_blocks - 1 ? n_samples : (b + 1) * block_size
        test_idx = indices[test_start:test_end]

        push!(splits, (train_idx, test_idx))
    end

    return splits
end

n_splits(cv::ProgressiveBlockSplit) = cv.n_blocks - cv.min_train_blocks


"""
    get_cross_validator(strategy::Symbol; n_samples=nothing, n_folds=5, kwargs...)

Factory function to get appropriate cross-validator.

# Arguments
- `strategy`: :blocked, :rolling, :panel, or :progressive
- `n_samples`: Number of samples (required for :rolling)
- `n_folds`: Number of folds
"""
function get_cross_validator(strategy::Symbol; n_samples::Union{Int,Nothing}=nothing, n_folds::Int=5, kwargs...)
    if strategy == :blocked
        return BlockedTimeSeriesSplit(; n_splits=n_folds, kwargs...)
    elseif strategy == :rolling
        isnothing(n_samples) && error("n_samples required for rolling strategy")
        initial_window = n_samples ÷ 2
        horizon = (n_samples - initial_window) ÷ n_folds
        step = horizon
        return RollingOriginSplit(; initial_window=initial_window, step=step, horizon=horizon, kwargs...)
    elseif strategy == :panel
        return PanelStratifiedSplit(; n_splits=n_folds, kwargs...)
    elseif strategy == :progressive
        return ProgressiveBlockSplit(; n_blocks=n_folds * 2, min_train_blocks=n_folds, kwargs...)
    else
        error("Unknown strategy: $strategy. Choose from: :blocked, :rolling, :panel, :progressive")
    end
end

# Import Random for shuffle
using Random

end  # module CrossFitting
