# run_hls_2024.tcl - Versión corregida

#==============================================================================
# CONFIGURACION DEL PROYECTO
#==============================================================================
open_project -reset bfp_proj_opt
set_top bfp_encode_top

#==============================================================================
# ARCHIVOS FUENTE
#==============================================================================
add_files bfp_hls.h
add_files bfp_ops_hls.h
add_files bfp_top_hls.cpp

#==============================================================================
# Testbench para C simulation
#==============================================================================
add_files -tb bfp_hls_tb.cc

#==============================================================================
# CONFIGURACION DE LA SOLUCION
#==============================================================================
open_solution -reset "sol1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 200MHz -name default 

#==============================================================================
# CONFIGURACION DE COMPILACION
#==============================================================================
config_compile -name_max_length 80
config_dataflow -strict_mode warning
config_rtl -deadlock_detection sim
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64 
config_interface -m_axi_auto_max_ports=0
config_export -format xo -ipname bfp_encode_top

#==============================================================================
# EJECUCION COMPLETA DEL FLUJO
#==============================================================================
puts "\n=========================================="
puts "Starting C Simulation (csim)"
puts "==========================================\n"

# Ejecutar simulación C del testbench
csim_design

puts "\n=========================================="
puts "Starting C Synthesis (csynth)"
puts "Configuration: WE=5, WM=7, N=16"
puts "==========================================\n"

csynth_design

puts "\n=========================================="
puts "Exporting RTL design"
puts "==========================================\n"

export_design -format ip_catalog

#==============================================================================
# REPORTE FINAL
#==============================================================================
puts "\n=========================================="
puts "HLS Flow Complete!"
puts "=========================================="
puts "Project: bfp_proj_opt"
puts "Check reports in: bfp_proj_opt/sol1/syn/report/"
puts "Testbench results should be visible above"
puts "\n"

exit
