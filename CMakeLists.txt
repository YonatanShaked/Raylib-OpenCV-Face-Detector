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

add_executable(rl_face_tracker
  src/main.cpp
  src/face_cv.cpp
  src/camera_handler.cpp
  src/raylib_utils.cpp
)

target_include_directories(rl_face_tracker PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

if (raylib_FOUND)
  target_link_libraries(rl_face_tracker PRIVATE raylib ${OpenCV_LIBS})
else()
  target_include_directories(rl_face_tracker PRIVATE ${RAYLIB_INCLUDE_DIRS})
  target_link_directories(rl_face_tracker PRIVATE ${RAYLIB_LIBRARY_DIRS})
  target_link_libraries(rl_face_tracker PRIVATE ${RAYLIB_LIBRARIES} ${OpenCV_LIBS})
endif()

if (UNIX)
  target_link_libraries(rl_face_tracker PRIVATE m pthread dl)
endif()

set(RLFT_ASSETS
  assets/haarcascade_frontalface_default.xml
  assets/lbfmodel.yaml
  assets/hat.obj
  assets/head.obj
  assets/glasses.obj
  assets/shaders/lighting.vs
  assets/shaders/lighting.fs
)

foreach(f IN LISTS RLFT_ASSETS)
  add_custom_command(TARGET rl_face_tracker POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:rl_face_tracker>/$<PATH:GET_PARENT_PATH,${f}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_SOURCE_DIR}/${f}
      $<TARGET_FILE_DIR:rl_face_tracker>/${f}
  )
endforeach()
