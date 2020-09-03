

set (QuickTutorial_PLUGIN_TESTS_SOURCES
        catch.hpp
        TestsMain.cc
        IntegrationTests.cc
        # Add component tests here
        )

add_executable(QuickTutorial_plugin_tests ${QuickTutorial_PLUGIN_TESTS_SOURCES})

find_package(JANA REQUIRED)

target_include_directories(QuickTutorial_plugin_tests PUBLIC ../src)
target_include_directories(QuickTutorial_plugin_tests PUBLIC ${JANA_INCLUDE_DIR})

target_link_libraries(QuickTutorial_plugin_tests QuickTutorial_plugin)
target_link_libraries(QuickTutorial_plugin_tests ${JANA_LIBRARY})

install(TARGETS QuickTutorial_plugin_tests DESTINATION bin)

