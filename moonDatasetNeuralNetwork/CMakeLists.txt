# CMakeList.txt : CMake project for moonDatasetNeuralNetwork, include source and define
# project specific logic here.
#
find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)
find_path(RAPIDCSV_INCLUDE_DIRS "rapidcsv.h")
# Add source to this project's executable.
add_executable (moonDatasetNeuralNetwork "main.cpp" "Forward.cpp" "Activation.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET moonDatasetNeuralNetwork PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-system sfml-network sfml-graphics sfml-window)
target_include_directories(${PROJECT_NAME} PRIVATE ${RAPIDCSV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE Resources)

set_target_properties(
	${PROJECT_NAME} PROPERTIES
	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)