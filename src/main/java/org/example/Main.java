package org.example;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

public class Main {
	public static void main(String[] args) {
		int dimension = 1024;
		Device device = Device.of("mps", 0);
		//Device device = Device.cpu();
		System.out.println(device.isGpu());
		try (NDManager manager = NDManager.newBaseManager(device)) {
			NDArray array1 = manager.randomUniform(0, 1, new Shape(dimension, dimension));
			NDArray array2 = manager.randomUniform(0, 1, new Shape(dimension, dimension));
			NDArray result = array1.add(array2).mul(10).matMul(array1.transpose()).div(5);
			System.out.println(result);
		}
	}
}