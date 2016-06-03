 #
 # @brief Samu C. Turing
 #
 # @file CMakeLists.txt
 # @author  Norbert Bátfai <nbatfai@gmail.com>
 # @version 0.0.1
 #
 # @section LICENSE
 #
 # Copyright (C) 2015 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 #
 # @section DESCRIPTION
 #
 # 
 
cmake_minimum_required(VERSION 2.8.5)
project(SAMUCTURING)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")

add_definitions(-DLIFEOFGAME)
add_definitions(-DQ_LOOKUP_TABLE)

add_custom_target(
    .
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
)

add_executable(SamuCTuring orchmach1.hpp SamuQl.h GameOfLife.cpp GameOfLife.h SamuBrain.cpp SamuBrain.h  main.cpp  )

