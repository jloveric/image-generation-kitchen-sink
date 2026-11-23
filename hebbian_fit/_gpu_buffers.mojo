from std.python import PythonObject
from max.gpu.host import DeviceBuffer
from layout import TileTensor, row_major

comptime Layout = type_of(row_major((Int(0),)))
comptime Read = TileTensor[DType.float32, Layout, ImmutAnyOrigin]
comptime Write = TileTensor[DType.float32, Layout, MutAnyOrigin]
comptime Buffer = DeviceBuffer[DType.float32]
comptime Block = 128


def read(buffer: Buffer) -> Read:
    return TileTensor(buffer, row_major((len(buffer),))).as_imm()


def write(mut buffer: Buffer) -> Write:
    return TileTensor(buffer, row_major((len(buffer),)))


def array_pointer(array: PythonObject) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=array.ctypes.data))


def upload(mut buffer: Buffer, array: PythonObject, count: Int) raises:
    var source = array_pointer(array)
    with buffer.map_to_host() as host:
        for i in range(count):
            host[i] = source[unsafe_offset=i]


def download(mut buffer: Buffer, array: PythonObject, count: Int) raises:
    var destination = array_pointer(array)
    with buffer.map_to_host() as host:
        for i in range(count):
            destination[unsafe_offset=i] = host[i]


