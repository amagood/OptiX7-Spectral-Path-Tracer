# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_policy(SET CMP0015 NEW)

include_directories("../imgui")
include_directories("../GL")
include_directories("../imgui/backends")
include_directories("..")
link_directories("../lib")

set(imgui
        ../imgui/backends/imgui_impl_glfw.cpp
        ../imgui/backends/imgui_impl_opengl3.cpp
        ../imgui/backends/imgui_impl_glfw.h
        ../imgui/backends/imgui_impl_opengl3.h
        ../imgui/imgui.cpp
        ../imgui/imgui_draw.cpp
        ../imgui/imgui_demo.cpp
        ../imgui/imgui_tables.cpp
        ../imgui/imgui_widgets.cpp

        )


add_library(glfWindow
  GLFWindow.h
  GLFWindow.cpp
  ${imgui}
  )
