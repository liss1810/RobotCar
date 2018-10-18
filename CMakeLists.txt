#cmake要求最小版本号
cmake_minimum_required(VERSION 3.2)
#指定项目名称
project(RobotCarDataStitch)

#c11
set(CMAKE_CXX_STANDARD 11)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
include_directories(${PROJECT_SOURCE_DIR}/src)
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )


#Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} )

#opencv
find_package(OpenCV REQUIRED)
list(APPEND 3RD_LIBS ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS} )

#Qt4
FIND_PACKAGE(Qt4 COMPONENTS QtCore QtXml QtOpenGL QtGui QtNetwork REQUIRED)
INCLUDE(${QT_USE_FILE})
list(APPEND 3RD_LIBS ${QT_LIBRARIES})

#COMMON_GQ
include_directories(${PROJECT_SOURCE_DIR}/src/Common_GQ)
file(GLOB COMMON_GQ_FILES ${PROJECT_SOURCE_DIR}/src/Common_GQ/*.cpp)

#apriltags
find_package(apriltags)
include_directories(${apriltags_INCLUDE_DIRS})

#PanoStitch
file(GLOB PanoStitch_FILES ${PROJECT_SOURCE_DIR}/src/PanoStitch/*.* )
add_library(PanoStitch ${PanoStitch_FILES} ${COMMON_GQ_FILES})
target_link_libraries(PanoStitch ${3RD_LIBS})

#CornerFinder
file(GLOB CornerFinder_FILES ${PROJECT_SOURCE_DIR}/src/CornerFinder/*.* )
add_library(CornerFinder ${CornerFinder_FILES} ${COMMON_GQ_FILES})
target_link_libraries(CornerFinder ${3RD_LIBS})

#pano13
add_subdirectory(libpano13-2.9.19)
include_directories(libpano13-2.9.19)
target_link_libraries(PanoStitch pano13)

FIND_PACKAGE(TIFF REQUIRED)
INCLUDE_DIRECTORIES(${TIFF_INCLUDE_DIR})
FIND_PACKAGE(JPEG REQUIRED)
INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
FIND_PACKAGE(PNG REQUIRED)
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIR})
target_link_libraries(PanoStitch ${TIFF_LIBRARIES} ${ZLIB_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})
target_link_libraries(PanoStitch ${apriltags_LIBRARIES})
#源文件
#set(SOURCE_FILES src/main.cpp)
#test exe
file(GLOB TEST_FILES ${PROJECT_SOURCE_DIR}/src/test/*.*) 
add_executable(${PROJECT_NAME}_TEST ${TEST_FILES} ${COMMON_GQ_FILES})
target_link_libraries(${PROJECT_NAME}_TEST  PanoStitch CornerFinder)

set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
foreach(CompilerFlag ${CompilerFlags})
   string(REPLACE "/MT" "/MD" ${CompilerFlag} "${${CompilerFlag}}")
endforeach()

set(CMAKE_EXE_LINKER_FLAGS "")

#编译可执行文件
#add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_GQ_FILES})
#可执行文件所需要的库

