using Test
using SystemDynamicsBuildR.unit_func
using Unitful

@testset "unit_func tests" begin

    @testset "convert_u - Quantity to Quantity" begin
        @testset "Length conversions" begin
            # Meters to kilometers
            @test convert_u(1000.0u"m", 1.0u"km") == 1.0u"km"
            @test convert_u(500.0u"m", 1.0u"km") == 0.5u"km"
            
            # Centimeters to meters
            @test convert_u(100.0u"cm", 1.0u"m") == 1.0u"m"
            @test convert_u(250.0u"cm", 1.0u"m") == 2.5u"m"
            
            # Feet to meters
            @test convert_u(3.28084u"ft", 1.0u"m") ≈ 1.0u"m" atol=0.0001u"m"
        end

        @testset "Mass conversions" begin
            # Grams to kilograms
            @test convert_u(1000.0u"g", 1.0u"kg") == 1.0u"kg"
            @test convert_u(500.0u"g", 1.0u"kg") == 0.5u"kg"
            
            # Kilograms to grams
            @test convert_u(2.5u"kg", 1.0u"g") == 2500.0u"g"
        end

        @testset "Time conversions" begin
            # Seconds to hours
            @test convert_u(3600.0u"s", 1.0u"hr") == 1.0u"hr"
            @test convert_u(7200.0u"s", 1.0u"hr") == 2.0u"hr"
            
            # Hours to seconds
            @test convert_u(2.0u"hr", 1.0u"s") == 7200.0u"s"
            
            # Minutes to seconds
            @test convert_u(5.0u"minute", 1.0u"s") == 300.0u"s"
        end

        @testset "Temperature conversions" begin
            # Celsius to Kelvin (note: temperature conversions are affine)
            result = convert_u(0.0u"°C", 1.0u"K")
            @test result ≈ 273.15u"K" atol=0.01u"K"
        end

        @testset "No conversion when units match" begin
            x = 5.0u"m"
            result = convert_u(x, 1.0u"m")
            @test result === x  # Should return the same object
            @test result == 5.0u"m"
            
            y = 10.0u"kg"
            result = convert_u(y, 2.0u"kg")
            @test result === y
        end
    end

    @testset "convert_u - Quantity to Units" begin
        @testset "Basic conversions" begin
            @test convert_u(100.0u"cm", u"m") == 1.0u"m"
            @test convert_u(1000.0u"g", u"kg") == 1.0u"kg"
            @test convert_u(60.0u"s", u"minute") == 1.0u"minute"
        end

        @testset "Complex units" begin
            # Speed: m/s to km/hr
            @test convert_u(1.0u"m/s", u"km/hr") ≈ 3.6u"km/hr" atol=0.01u"km/hr"
            
            # Area: cm² to m²
            @test convert_u(10000.0u"cm^2", u"m^2") == 1.0u"m^2"
        end

        @testset "No conversion when units match" begin
            x = 7.5u"m"
            result = convert_u(x, u"m")
            @test result === x
            @test result == 7.5u"m"
        end
    end

    @testset "convert_u - Real to Quantity (attach units)" begin
        @testset "Attach length units" begin
            @test convert_u(5.0, 1.0u"m") == 5.0u"m"
            @test convert_u(10.0, 2.5u"km") == 10.0u"km"
            @test convert_u(3.14, 1.0u"cm") == 3.14u"cm"
        end

        @testset "Attach mass units" begin
            @test convert_u(100.0, 1.0u"kg") == 100.0u"kg"
            @test convert_u(50, 1.0u"g") == 50.0u"g"
        end

        @testset "Attach time units" begin
            @test convert_u(60.0, 1.0u"s") == 60.0u"s"
            @test convert_u(2.5, 1.0u"hr") == 2.5u"hr"
        end

        @testset "Works with integers" begin
            @test convert_u(10, 1.0u"m") == 10.0u"m"
            @test convert_u(5, 1.0u"kg") == 5.0u"kg"
        end

        @testset "Works with different numeric types" begin
            @test convert_u(Int32(5), 1.0u"m") == 5.0u"m"
            @test convert_u(Float32(3.5), 1.0u"kg") == 3.5u"kg"
        end
    end

    @testset "convert_u - Real to Units (attach units)" begin
        @testset "Attach various units" begin
            @test convert_u(100.0, u"m") == 100.0u"m"
            @test convert_u(5.0, u"kg") == 5.0u"kg"
            @test convert_u(2.5, u"s") == 2.5u"s"
            @test convert_u(10, u"A") == 10.0u"A"
        end

        @testset "Complex units" begin
            @test convert_u(50.0, u"m/s") == 50.0u"m/s"
            @test convert_u(9.8, u"m/s^2") == 9.8u"m/s^2"
        end
    end

    @testset "convert_u - Array conversions" begin
        @testset "Array of Quantities - convert units" begin
            arr = [100.0u"cm", 200.0u"cm", 300.0u"cm"]
            result = convert_u(arr, u"m")
            @test result == [1.0u"m", 2.0u"m", 3.0u"m"]
            
            arr2 = [1000.0u"g", 2000.0u"g", 3000.0u"g"]
            result2 = convert_u(arr2, 1.0u"kg")
            @test result2 == [1.0u"kg", 2.0u"kg", 3.0u"kg"]
        end

        @testset "Array of Quantities - no conversion needed" begin
            arr = [1.0u"m", 2.0u"m", 3.0u"m"]
            result = convert_u(arr, u"m")
            @test result == arr
            @test result == [1.0u"m", 2.0u"m", 3.0u"m"]
        end

        @testset "Array of Reals - attach units" begin
            arr = [1.0, 2.0, 3.0]
            result = convert_u(arr, u"m")
            @test result == [1.0u"m", 2.0u"m", 3.0u"m"]
            
            arr2 = [10, 20, 30]
            result2 = convert_u(arr2, 1.0u"kg")
            @test result2 == [10.0u"kg", 20.0u"kg", 30.0u"kg"]
        end

        @testset "Empty arrays" begin
            empty_arr = Unitful.Quantity{Float64}[]
            result = convert_u(empty_arr, u"m")
            @test isempty(result)
            
            empty_reals = Float64[]
            result2 = convert_u(empty_reals, u"kg")
            @test isempty(result2)
        end

        @testset "Multi-dimensional arrays" begin
            mat = [1.0 2.0; 3.0 4.0]
            result = convert_u(mat, u"m")
            @test result == [1.0u"m" 2.0u"m"; 3.0u"m" 4.0u"m"]
            
            mat_units = [100.0u"cm" 200.0u"cm"; 300.0u"cm" 400.0u"cm"]
            result2 = convert_u(mat_units, u"m")
            @test result2 == [1.0u"m" 2.0u"m"; 3.0u"m" 4.0u"m"]
        end
    end

    @testset "Integration tests" begin
        @testset "Chaining conversions" begin
            # Start with unitless, add units, convert
            x = 1000.0
            x_with_units = convert_u(x, u"mm")
            x_converted = convert_u(x_with_units, u"m")
            @test x_converted == 1.0u"m"
        end

        @testset "Mixed operations" begin
            # Create quantities and convert between them
            distance = convert_u(5000.0, u"m")
            time = convert_u(3600.0, u"s")
            
            # Convert distance to km
            distance_km = convert_u(distance, u"km")
            @test distance_km == 5.0u"km"
            
            # Convert time to hours
            time_hr = convert_u(time, u"hr")
            @test time_hr == 1.0u"hr"
            
            # Speed (this tests that our conversions preserve dimensional analysis)
            speed = distance / time
            @test Unitful.dimension(speed) == Unitful.dimension(u"m/s")
        end

        @testset "Array operations" begin
            # Create array with units, convert, then do array operations
            distances = convert_u([100.0, 200.0, 300.0], u"cm")
            distances_m = convert_u(distances, u"m")
            
            @test sum(distances_m) == 6.0u"m"
            @test maximum(distances_m) == 3.0u"m"
            @test minimum(distances_m) == 1.0u"m"
        end

        @testset "Preserving precision" begin
            # Test that conversions don't introduce unnecessary rounding errors
            x = 1.23456789u"m"
            y = convert_u(x, u"cm")
            z = convert_u(y, u"m")
            @test z ≈ x atol=1e-10u"m"
        end
    end

    @testset "Edge cases" begin
        @testset "Zero values" begin
            @test convert_u(0.0u"m", u"km") == 0.0u"km"
            @test convert_u(0.0, u"m") == 0.0u"m"
            @test convert_u([0.0, 0.0], u"kg") == [0.0u"kg", 0.0u"kg"]
        end

        @testset "Negative values" begin
            @test convert_u(-5.0u"m", u"cm") == -500.0u"cm"
            @test convert_u(-10.0, u"kg") == -10.0u"kg"
            @test convert_u([-1.0, -2.0], u"m") == [-1.0u"m", -2.0u"m"]
        end

        @testset "Very large and very small values" begin
            @test convert_u(1e10u"m", u"km") == 1e7u"km"
            @test convert_u(1e-10u"m", u"nm") ≈ 0.1u"nm" rtol=1e-10
        end
    end

end