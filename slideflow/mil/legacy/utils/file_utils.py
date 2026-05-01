# Slideflow-GPL - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-GPL.
#
# Slideflow-GPL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Slideflow-GPL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Slideflow-GPL. If not, see <https://www.gnu.org/licenses/>.
#
# This file incorporates work from CLAM, which is licensed
# under the GNU General Public License, Version 3. The original
# license and source code can be found at https://github.com/mahmoodlab/CLAM.

import pickle


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file
