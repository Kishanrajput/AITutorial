

set (QuickTutorial_PLUGIN_SOURCES
		QuickTutorial.cc
		QuickTutorialProcessor.cc
		QuickTutorialProcessor.h
		Model.cpp
		Tensor.cpp
		RandomSource.cc
		RandomSource.h
	)


add_library(QuickTutorial_plugin SHARED ${QuickTutorial_PLUGIN_SOURCES})

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/tf/lib)
#add_executable(QuickTutorial_plugin Model.cpp Tensor.cpp)
target_include_directories(QuickTutorial_plugin PRIVATE ../include $ENV{HOME}/tf/include)
target_link_libraries (QuickTutorial_plugin "${TENSORFLOW_LIB}")


#find_package(OpenCV REQUIRED )
#target_link_libraries(QuickTutorial_plugin ${OpenCV_LIBS} )

find_package(JANA REQUIRED)
target_include_directories(QuickTutorial_plugin PUBLIC ${JANA_INCLUDE_DIR})
target_link_libraries(QuickTutorial_plugin ${JANA_LIBRARY})
set_target_properties(QuickTutorial_plugin PROPERTIES PREFIX "" OUTPUT_NAME "QuickTutorial" SUFFIX ".so")
install(TARGETS QuickTutorial_plugin DESTINATION plugins)

