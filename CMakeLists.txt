cmake_minimum_required(VERSION 3.20)
project(raylib_opencv_facemark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_package(raylib QUIET)
if (NOT raylib_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(RAYLIB REQUIRED raylib)
endif()

add_executable(app
  src/main.cpp
  src/face_cv.cpp
  src/camera_handler.cpp
)

target_include_directories(app PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

if (raylib_FOUND)
  target_link_libraries(app PRIVATE raylib ${OpenCV_LIBS})
else()
  target_include_directories(app PRIVATE ${RAYLIB_INCLUDE_DIRS})
  target_link_directories(app PRIVATE ${RAYLIB_LIBRARY_DIRS})
  target_link_libraries(app PRIVATE ${RAYLIB_LIBRARIES} ${OpenCV_LIBS})
endif()

if (UNIX)
  target_link_libraries(app PRIVATE m pthread dl)
endif()

add_custom_command(TARGET app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:app>/assets
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/haarcascade_frontalface_default.xml
    $<TARGET_FILE_DIR:app>/assets/haarcascade_frontalface_default.xml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/lbfmodel.yaml
    $<TARGET_FILE_DIR:app>/assets/lbfmodel.yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/hat.obj
    $<TARGET_FILE_DIR:app>/assets/hat.obj
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/head.obj
    $<TARGET_FILE_DIR:app>/assets/head.obj
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/glasses.obj
    $<TARGET_FILE_DIR:app>/assets/glasses.obj
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:app>/assets/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/shaders/lighting.vs
    $<TARGET_FILE_DIR:app>/assets/shaders/lighting.vs
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/assets/shaders/lighting.fs
    $<TARGET_FILE_DIR:app>/assets/shaders/lighting.fs
)
