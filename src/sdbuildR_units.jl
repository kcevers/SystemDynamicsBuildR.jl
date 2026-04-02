"""
    sdbuildR_units

Custom unit definitions for system dynamics modeling.

This module extends Unitful with domain-specific units commonly used in
system dynamics, economics, and scientific modeling:

# Time Units
- `common_yr`: Common year (365 days exactly, no leap years)
- `common_quarter`: Quarter of common year (365/4 days)
- `common_month`: Month of common year (365/12 days)
- `quarter`: Quarter of Julian year (1/4 year)
- `month`: Month of Julian year (1/12 year)

# Volume Units
- `quart`: US liquid quart (946.35 cm³)
- `US_gal`: US gallon (3.785411784 L)
- `fluidOunce`: US fluid ounce (29.5735 mL)

# Mass Units
- `tonne`: Metric ton (1000 kg)
- `ton`: US short ton (907.18474 kg)

# Amount Units
- `atom`: Single atom (1/Avogadro's number mol)
- `molecule`: Single molecule (1/Avogadro's number mol)

# Currency Units
- `EUR`: Euro
- `USD`: US Dollar
- `GBP`: British Pound
(Note: These are dimensionless units for accounting purposes)

# Angular Units
- `deg_`: Degree (π/180 radians)

# Electromagnetic Units
- `ohm_`: Ohm (electrical resistance, V/A)

# Physical Constants (as units)
- `reduced_Planck_constant`: ℏ = h/(2π)
- `superconducting_magnetic_flux_quantum`: Φ₀ = h/(2e)
- `Stefan_Boltzmann_constant`: σ
- `Bohr_magneton`: μ_B
- `Rydberg_constant`: R_∞
- `magnetic_constant`: μ₀
- `electric_constant`: ε₀
- `anghertz`: Angular frequency unit (2π/s)

# Temperature Units
- `degF`: Degree Fahrenheit
- `degC`: Degree Celsius

# Usage
```julia
using Unitful
using SystemDynamicsBuildR.sdbuildR_units

# Time periods
duration = 2.5u"common_yr"
quarterly_rate = 0.05u"1/quarter"

# Volumes
volume = 5.0u"US_gal"
amount = 32.0u"fluidOunce"

# Mass
weight = 2.5u"ton"  # US ton
mass = 1.5u"tonne"  # Metric tonne

# Currency (dimensionless)
price = 100.0u"USD"
cost = 75.0u"EUR"

# Molecular quantities
particles = 1.0e23u"molecule"
```

# Notes
- Common time units (common_yr, common_quarter, common_month) use 365-day years
- Standard time units (quarter, month) use Julian years (365.25 days)
- Currency units are dimensionless (for accounting, not exchange rates)
- Physical constants are provided as units for convenience in calculations
"""
module sdbuildR_units
    using Unitful
    
    # Group 1: Units with no dependencies on other custom units
    @unit common_yr "common_yr" CommonYear 365u"d" false
    @unit common_quarter "common_quarter" CommonQuarter 365//4*u"d" false
    @unit common_month "common_month" CommonMonth 365//12*u"d" false
    @unit quarter "quarter" Quarter 1//4*u"yr" false
    @unit month "month" Month 1//12*u"yr" false
    @unit quart "quart" Quart 946.35u"cm^3" false
    @unit US_gal "US_gal" USGallon 0.003785411784u"m^3" false
    @unit fluidOunce "fluidOunce" FluidOunce 29.5735295625u"mL" false
    @unit tonne "tonne" Tonne 1000u"kg" false
    @unit ton "ton" Ton 907.18474u"kg" false
    @unit atom "atom" Atom 1/6.02214076e23*u"mol" false
    @unit molecule "molecule" Molecule 1/6.02214076e23*u"mol" false
    @unit EUR "EUR" Euro 1 false
    @unit USD "USD" USDollar 1 false
    @unit GBP "GBP" BritishPound 1 false
    @unit deg_ "deg_" Degree_ π/180 false
    @unit ohm_ "ohm_" Ohm_ 1u"V"/u"A" false
    @unit superconducting_magnetic_flux_quantum "superconducting_magnetic_flux_quantum" MagneticFluxQuantum u"h"/(2u"q") false
    # @unit degF "degF" DegreeFahrenheit 45967//100*u"Ra" false # Added in Unitful v1.26.0
    # @unit degC "degC" DegreeCelsius 27315//100*u"K" false # Added in Unitful v1.26.0
    @unit anghertz "anghertz" AngularHertz 2π/u"s" false
    @unit Rydberg_constant "Rydberg_constant" RydbergConstant 10_973_731.568_160/u"m" false
    
    # Register reduced_Planck_constant before using it
    @unit reduced_Planck_constant "reduced_Planck_constant" ReducedPlanckConstant u"h"/(2π) false
    Unitful.register(sdbuildR_units)
    
    # Now can use reduced_Planck_constant in definitions
    @unit Stefan_Boltzmann_constant "Stefan_Boltzmann_constant" StefanBoltzmannConstant π^2*u"k"^4/(60*u"reduced_Planck_constant"^3*u"c"^2) false
    @unit Bohr_magneton "Bohr_magneton" BohrMagneton u"q"*u"reduced_Planck_constant"/(2*u"me") false
    Unitful.register(sdbuildR_units)
    
    # Register magnetic_constant before using it
    @unit magnetic_constant "magnetic_constant" MagneticConstant 4π*(1//10)^7*u"H"/u"m" false
    Unitful.register(sdbuildR_units)
    
    # Now can use magnetic_constant (via μ0 alias)
    @unit electric_constant "electric_constant" ElectricConstant 1/(u"μ0"*u"c"^2) false
    
    # Final registration for all remaining units
    Unitful.register(sdbuildR_units)
    
    function __init__()
        Unitful.register(sdbuildR_units)
    end
    
end # module
