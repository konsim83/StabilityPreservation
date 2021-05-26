###############################################################################
###############################################################################
#
# Include directory for sources
#
include_directories(${MsFEC_INCLUDE_DIR})
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Name all sources
#
#file(GLOB_RECURSE MsFEC_TARGET_LIB_SRC  "*.cc") # source files
#set(MsFEC_TARGET_LIB_SRC ${MsFEC_TARGET_LIB_SRC})

set(MsFEC_OTHER_TOOLS_LIBRARY_SRC
	my_other_tools.cc)

set(MsFEC_EQN_DATA_LIBRARY_SRC
	equation_data/eqn_boundary_vals.cc
	equation_data/eqn_coeff_A.cc
	equation_data/eqn_coeff_B.cc
	equation_data/eqn_coeff_R.cc
	equation_data/eqn_exact_solution_lin.cc
	equation_data/eqn_rhs.cc)
	
set(MsFEC_SHAPE_FUNCTIONS_LIBRARY_SRC
	functions/basis_q1.inst.cc
	functions/basis_q1_grad.inst.cc
	functions/basis_nedelec.inst.cc
	functions/basis_nedelec_curl.inst.cc
	functions/basis_raviart_thomas.inst.cc
	functions/my_mapping_q1.inst.cc)
	
set(MsFEC_VECTOR_TOOLS_LIBRARY_SRC
	vector_tools/my_vector_tools.inst.cc)

set(MsFEC_Q_LIBRARY_SRC	
	Q/q_parameters.cc
	Q/q_post_processor.cc
	Q/q_ref.cc
	Q/q_basis.cc
	Q/q_global.cc)

set(MsFEC_Q_NED_LIBRARY_SRC	
	Q_Ned/q_ned_basis.cc
	Q_Ned/q_ned_global.cc
	Q_Ned/q_ned_post_processor.cc
	Q_Ned/q_ned_ref.cc
	Q_Ned/q_ned_parameters.cc)
		
set(MsFEC_NED_RT_LIBRARY_SRC	
	Ned_RT/ned_rt_basis.cc
	Ned_RT/ned_rt_global.cc
	Ned_RT/ned_rt_post_processor.cc
	Ned_RT/ned_rt_ref.cc
	Ned_RT/ned_rt_parameters.cc)
	
set(MsFEC_RT_DQ_LIBRARY_SRC	
	RT_DQ/rt_dq_basis.cc
	RT_DQ/rt_dq_global.cc
	RT_DQ/rt_dq_ref.cc
	RT_DQ/rt_dq_post_processor.cc
	RT_DQ/rt_dq_parameters.cc)	


print_all_args (
	${MsFEC_EQN_DATA_LIBRARY_SRC}
)

print_all_args (
	${MsFEC_SHAPE_FUNCTIONS_LIBRARY_SRC}
)

print_all_args (
	${MsFEC_VECTOR_TOOLS_LIBRARY_SRC}
)

print_all_args (
	${MsFEC_Q_LIBRARY_SRC}
)

print_all_args (
	${MsFEC_Q_NED_LIBRARY_SRC}
)

print_all_args (
	${MsFEC_NED_RT_LIBRARY_SRC}
)

print_all_args (
	${MsFEC_RT_DQ_LIBRARY_SRC}
)
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Compile and link the sources as SHARED
#

add_library (MsFEC_OTHER_TOOLS_LIBRARY SHARED ${MsFEC_OTHER_TOOLS_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_OTHER_TOOLS_LIBRARY)

add_library (MsFEC_EQN_DATA_LIBRARY SHARED ${MsFEC_EQN_DATA_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_EQN_DATA_LIBRARY)

add_library (MsFEC_SHAPE_FUNCTIONS_LIBRARY SHARED ${MsFEC_SHAPE_FUNCTIONS_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_SHAPE_FUNCTIONS_LIBRARY)

add_library (MsFEC_VECTOR_TOOLS_LIBRARY SHARED ${MsFEC_VECTOR_TOOLS_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_VECTOR_TOOLS_LIBRARY)

add_library (MsFEC_Q_LIBRARY SHARED ${MsFEC_Q_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_Q_LIBRARY)
TARGET_LINK_LIBRARIES(MsFEC_Q_LIBRARY
	MsFEC_EQN_DATA_LIBRARY
	MsFEC_SHAPE_FUNCTIONS_LIBRARY
	MsFEC_VECTOR_TOOLS_LIBRARY
	MsFEC_OTHER_TOOLS_LIBRARY)

add_library (MsFEC_Q_NED_LIBRARY SHARED ${MsFEC_Q_NED_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_Q_NED_LIBRARY)
TARGET_LINK_LIBRARIES(MsFEC_Q_NED_LIBRARY
	MsFEC_EQN_DATA_LIBRARY
	MsFEC_VECTOR_TOOLS_LIBRARY
	MsFEC_OTHER_TOOLS_LIBRARY
	MsFEC_SHAPE_FUNCTIONS_LIBRARY)

add_library (MsFEC_NED_RT_LIBRARY SHARED ${MsFEC_NED_RT_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_NED_RT_LIBRARY)
TARGET_LINK_LIBRARIES(MsFEC_NED_RT_LIBRARY 
	MsFEC_EQN_DATA_LIBRARY
	MsFEC_VECTOR_TOOLS_LIBRARY
	MsFEC_OTHER_TOOLS_LIBRARY
	MsFEC_SHAPE_FUNCTIONS_LIBRARY)

add_library (MsFEC_RT_DQ_LIBRARY SHARED ${MsFEC_RT_DQ_LIBRARY_SRC})
DEAL_II_SETUP_TARGET(MsFEC_RT_DQ_LIBRARY)
TARGET_LINK_LIBRARIES(MsFEC_RT_DQ_LIBRARY 
	MsFEC_EQN_DATA_LIBRARY
	MsFEC_VECTOR_TOOLS_LIBRARY
	MsFEC_OTHER_TOOLS_LIBRARY
	MsFEC_SHAPE_FUNCTIONS_LIBRARY)

#
# Install into the DESTINATION provided by CMAKE_INSTALL_PREFIX
#
#install (TARGETS ${MsFEC_LIBRARY} DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(MsFEC_Q "main_q.cxx")
DEAL_II_SETUP_TARGET(MsFEC_Q)
TARGET_LINK_LIBRARIES(MsFEC_Q
	MsFEC_Q_LIBRARY)

add_executable(MsFEC_Q_Ned "main_q_ned.cxx")
DEAL_II_SETUP_TARGET(MsFEC_Q_Ned)
TARGET_LINK_LIBRARIES(MsFEC_Q_Ned
	MsFEC_Q_NED_LIBRARY)

add_executable(MsFEC_Ned_RT "main_ned_rt.cxx")
DEAL_II_SETUP_TARGET(MsFEC_Ned_RT)
TARGET_LINK_LIBRARIES(MsFEC_Ned_RT 
	MsFEC_NED_RT_LIBRARY)

add_executable(MsFEC_RT_DQ "main_rt_dq.cxx")
DEAL_II_SETUP_TARGET(MsFEC_RT_DQ)
TARGET_LINK_LIBRARIES(MsFEC_RT_DQ 
	MsFEC_RT_DQ_LIBRARY)
###############################################################################
###############################################################################
