"""
    unit_func

Utilities for working with Unitful quantities, providing flexible unit conversion
and attachment for both unitful and unitless values.

This module simplifies unit handling in system dynamics models by:
- Converting between compatible units automatically
- Attaching units to unitless values
- Handling both scalar and array inputs
- Optimizing by skipping unnecessary conversions
"""
module unit_func

using Unitful

export convert_u

# ============================================================================
# Unit Conversion Functions
# ============================================================================

"""
    convert_u(x::Unitful.Quantity, unit_def::Unitful.Quantity)

Convert a unitful quantity `x` to match the units of `unit_def`.

If `x` already has the same units as `unit_def`, no conversion is performed
for efficiency. Otherwise, the value is converted to the target units.

# Arguments
- `x::Unitful.Quantity`: Value with units to convert
- `unit_def::Unitful.Quantity`: Template quantity whose units will be used

# Returns
- `Unitful.Quantity`: Converted value with units matching `unit_def`

# Examples
```julia
julia> convert_u(1000.0u"m", 1.0u"km")
1.0 km

julia> convert_u(5.0u"kg", 1.0u"g")
5000.0 g

julia> convert_u(2.0u"hr", 1.0u"s")
7200.0 s

# No conversion if units already match
julia> convert_u(5.0u"m", 1.0u"m")
5.0 m
```
"""
function convert_u(x::Unitful.Quantity, unit_def::Unitful.Quantity)
    if Unitful.unit(x) == Unitful.unit(unit_def)
        return x  # No conversion needed
    else
        return Unitful.uconvert(Unitful.unit(unit_def), x)
    end
end

"""
    convert_u(x::Unitful.Quantity, unit_def::Unitful.Units)

Convert a unitful quantity `x` to the specified units.

If `x` already has the specified units, no conversion is performed for efficiency.

# Arguments
- `x::Unitful.Quantity`: Value with units to convert
- `unit_def::Unitful.Units`: Target units (e.g., `u"m"`, `u"kg"`)

# Returns
- `Unitful.Quantity`: Converted value with the specified units

# Examples
```julia
julia> convert_u(100.0u"cm", u"m")
1.0 m

julia> convert_u(3600.0u"s", u"hr")
1.0 hr

julia> convert_u(1.0u"kg", u"g")
1000.0 g
```
"""
function convert_u(x::Unitful.Quantity, unit_def::Unitful.Units)
    if Unitful.unit(x) == unit_def
        return x  # No conversion needed
    else
        return Unitful.uconvert(unit_def, x)
    end
end

"""
    convert_u(x::Real, unit_def::Unitful.Quantity)

Attach units to a unitless number by using the units from `unit_def`.

This is useful when you have a raw number that should have units attached,
taking the unit type from an example quantity.

# Arguments
- `x::Real`: Unitless value
- `unit_def::Unitful.Quantity`: Template quantity whose units will be attached

# Returns
- `Unitful.Quantity`: Value `x` with units from `unit_def`

# Examples
```julia
julia> convert_u(5.0, 1.0u"m")
5.0 m

julia> convert_u(10, 2.5u"kg")
10.0 kg

julia> convert_u(3.14, 1.0u"rad")
3.14 rad
```
"""
function convert_u(x::Real, unit_def::Unitful.Quantity)
    return x * Unitful.unit(unit_def)
end

"""
    convert_u(x::Real, unit_def::Unitful.Units)

Attach units to a unitless number.

# Arguments
- `x::Real`: Unitless value
- `unit_def::Unitful.Units`: Units to attach (e.g., `u"m"`, `u"s"`)

# Returns
- `Unitful.Quantity`: Value `x` with the specified units

# Examples
```julia
julia> convert_u(100.0, u"m")
100.0 m

julia> convert_u(5, u"kg")
5.0 kg

julia> convert_u(2.5, u"s")
2.5 s
```
"""
function convert_u(x::Real, unit_def::Unitful.Units)
    return x * unit_def
end

"""
    convert_u(x::AbstractArray{<:Unitful.Quantity}, unit_def)

Convert an array of unitful quantities to match the target units.

Broadcasts the conversion over all elements of the array.

# Arguments
- `x::AbstractArray{<:Unitful.Quantity}`: Array of values with units
- `unit_def`: Target units (can be `Unitful.Quantity` or `Unitful.Units`)

# Returns
- `AbstractArray{<:Unitful.Quantity}`: Array with all elements converted

# Examples
```julia
julia> convert_u([1.0u"m", 2.0u"m", 3.0u"m"], u"cm")
3-element Vector{Quantity{Float64, 𝐋, Unitful.FreeUnits{(cm,), 𝐋, nothing}}}:
 100.0 cm
 200.0 cm
 300.0 cm

julia> convert_u([1000.0u"g", 2000.0u"g"], 1.0u"kg")
2-element Vector{Quantity{Float64, 𝐌, Unitful.FreeUnits{(kg,), 𝐌, nothing}}}:
 1.0 kg
 2.0 kg
```
"""
function convert_u(x::AbstractArray{<:Unitful.Quantity}, unit_def::Union{Unitful.Quantity, Unitful.Units})
    target_unit = unit_def isa Unitful.Quantity ? Unitful.unit(unit_def) : unit_def

    # Single pass conversion is faster than pre-checking all elements first.
    return Unitful.uconvert.(target_unit, x)
end

"""
    convert_u(x::AbstractArray{<:Real}, unit_def)

Attach units to an array of unitless numbers.

# Arguments
- `x::AbstractArray{<:Real}`: Array of unitless values
- `unit_def`: Units to attach (can be `Unitful.Quantity` or `Unitful.Units`)

# Returns
- `AbstractArray{<:Unitful.Quantity}`: Array with units attached to all elements

# Examples
```julia
julia> convert_u([1.0, 2.0, 3.0], u"m")
3-element Vector{Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}:
 1.0 m
 2.0 m
 3.0 m

julia> convert_u([10, 20, 30], 1.0u"kg")
3-element Vector{Quantity{Int64, 𝐌, Unitful.FreeUnits{(kg,), 𝐌, nothing}}}:
 10 kg
 20 kg
 30 kg
```
"""
function convert_u(x::AbstractArray{<:Real}, unit_def::Union{Unitful.Quantity, Unitful.Units})
    target_unit = unit_def isa Unitful.Quantity ? Unitful.unit(unit_def) : unit_def
    return x .* target_unit
end

end # module
