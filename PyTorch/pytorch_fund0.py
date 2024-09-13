import torch
print(torch.__version__)

# https://www.learnpytorch.io/00_pytorch_fundamentals/

# Scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)

# Get the Python number with a tensor (only works with one-element tensors)
print(scalar.item())

# Vector 
vector = torch.tensor([7, 7])
print(vector)

# Check number of dimensions of vector
print(vector.ndim)

# Check shape of vector
print(vector.shape)

# Matrix
MATRIX = torch.tensor([[7, 8],[9, 10]])

print(MATRIX)

# Check number of dimensions
print(MATRIX.ndim)

print(MATRIX.shape)

# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

print(TENSOR)

# Check the number of dimensions for TENSOR
print(TENSOR.ndim)

# Check shape of TENSOR
print(TENSOR.shape)

# Create a random tensor size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(random_tensor.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

# Craete a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, #defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded

print(float_32_tensor.shape)
print(float_32_tensor.dtype)
print(float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16)

print(float_16_tensor.dtype)

# Create a tensor 
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")


# Manipulating tensors

# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

# Multiply it by 10
print(tensor * 10)

# Tensors don't change unless reassigned!!!!!!
print(tensor)

# Subtract and reassign
tensor = tensor - 10
print(tensor)

# Add and reassign
tensor = tensor + 10
print(tensor)

# Can also use torch functions
print(torch.multiply(tensor, 10))
print(tensor)

# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

# Matrix multiplication
tensor = torch.tensor([1, 2, 3])
print(torch.matmul(tensor, tensor))

# Can also use "@" for matrix multiplication
print(tensor @ tensor)

#print(torch.cuda.is_available())

#device = "cuda" if torch.cuda.is_available() else "cpu"

#print(device)