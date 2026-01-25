import importlib.util
from types import ModuleType
from typing import Any, Generic, TypeVar

from typing_extensions import Self

ArrayLike = TypeVar("ArrayLike")


def _get_lib(*args: Any) -> ModuleType:
    libs = [a.__class__.__module__.split(".")[0] for a in args]
    if any(lib != libs[0] for lib in libs):
        raise ValueError(
            "args must be of the same library, " f"but got {', '.join(libs)}"
        )
    return importlib.import_module(libs[0])


class Quaternion(Generic[ArrayLike]):
    """Quaternion class."""

    _AXIS = 0
    np: ModuleType

    def __init__(
        self,
        *,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        w: ArrayLike,
        is_product_antimorphic: bool = True,
        eq_as_rotation: bool = True,
    ) -> None:
        """
        Quaternion class.

        Parameters
        ----------
        x : ArrayLike
            _description_
        y : ArrayLike
            _description_
        z : ArrayLike
            _description_
        w : ArrayLike
            _description_
        is_product_antimorphic : bool
            If True, Hamilton convention is used.
            If False, JPL convention is used. (Shuster's solution)

            JPL convention is often used in aerospace and robotics.
            Hamilton convention is more common in most other fields.
        eq_rotation : bool

            If True, the quaternion


        Following Unity's convention, W component is the last in the internal array.

        Breaking change from Unity's convention:
        - CamelCase to snake_case
        - Static methods are now class methods
        - removed Set...() and To...() public methods
        - removed this[int]
        - AngleAxis is now from_angle_axis()
        - identity is not class property but class function,
          as class properties are no longer supported in Python
        - lerp() and slerp() does not clamp by default
        - removed RotateTowards() as it is not a common operation,
          use self.lerp(other, self.lib.min(maxDegreesDelta / self.angle(other), 1)) instead


        References
        ----------
        Why and How to Avoid the Flipped Quaternion Multiplication
        https://arxiv.org/abs/1801.07478
        Handy Note for Quaternions
        https://www.mesw.co.jp/business/report/pdf/mss_18_07.pdf

        """
        self.np = _get_lib(x, y, z, w)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.q = self.np.stack([self.x, self.y, self.z, self.w], axis=self._AXIS)
        self.is_product_antimorphic = is_product_antimorphic
        self.eq_as_rotation = eq_as_rotation

    @classmethod
    def from_array(cls, q: ArrayLike, w_last: bool = True) -> Self:
        if w_last:
            return cls(
                x=q.take(0, axis=cls._AXIS),
                y=q.take(1, axis=cls._AXIS),
                z=q.take(2, axis=cls._AXIS),
                w=q.take(3, axis=cls._AXIS),
            )
        else:
            return cls(
                x=q.take(1, axis=cls._AXIS),
                y=q.take(2, axis=cls._AXIS),
                z=q.take(3, axis=cls._AXIS),
                w=q.take(0, axis=cls._AXIS),
            )

    @classmethod
    def from_array_xyz(cls, *, xyz: ArrayLike, w: ArrayLike) -> Self:
        return cls(
            x=xyz.take(0, axis=cls._AXIS),
            y=xyz.take(1, axis=cls._AXIS),
            z=xyz.take(2, axis=cls._AXIS),
            w=w,
        )

    @property
    def xyz(self) -> ArrayLike:
        return self.q[:3]

    @classmethod
    def identity(cls) -> Self:
        return cls(x=0, y=0, z=0, w=1)

    @property
    def norm(self) -> ArrayLike:
        return self.np.linalg.norm(self.q, axis=self._AXIS)

    @property
    def normalized(self) -> Self:
        return self.from_array(self.q / self.norm)

    @property
    def norm_xyz(self) -> ArrayLike:
        return self.np.linalg.norm(self.xyz, axis=self._AXIS)

    @property
    def normalized_xyz(self) -> ArrayLike:
        return self.xyz / self.norm_xyz

    def __getitem__(self, index: int) -> ArrayLike:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        elif index == 3:
            return self.w
        else:
            raise IndexError("Index out of range")

    def angle(self, other: "Quaternion | None" = None) -> ArrayLike:
        if other is None:
            return 2 * self.np.arccos(self.w)
        return self.np.arccos(self.dot(other))

    def axis(self) -> ArrayLike:
        angle = self.angle()
        return self.q[:3] / self.np.sin(self.angle / 2)

    def to_angle_axis(self) -> tuple[ArrayLike, ArrayLike]:
        angle = self.angle()
        axis = self.q[:3] / self.np.sin(angle / 2)
        return angle, axis

    @classmethod
    def from_angle_axis(cls, angle: ArrayLike, axis: ArrayLike) -> Self:
        lib = _get_lib(angle, axis)
        anglehalf = angle / 2
        return cls.from_array_xyz(xyz=axis * lib.sin(anglehalf), w=lib.cos(anglehalf))

    def dot(self, other: "Quaternion") -> ArrayLike:
        return self.np.sum(self.q * other.q, axis=self._AXIS)

    @classmethod
    def from_euler(
        cls, x: ArrayLike, y: ArrayLike, z: ArrayLike, *, convention: str
    ) -> Self:
        raise NotImplementedError

    def to_euler(self, *, convention: str) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        raise NotImplementedError

    def inverse(self) -> Self:
        return self.conjugate / self.np.sum(self.q**2)

    @property
    def conjugate(self) -> Self:
        return self.__class__(x=-self.x, y=-self.y, z=-self.z, w=self.w)

    def __add__(self, other: "Quaternion") -> Self:
        return self.__class__(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
            w=self.w + other.w,
        )

    def __sub__(self, other: "Quaternion") -> Self:
        return self.__add__(-other)

    def __neg__(self) -> Self:
        return self.__class__.from_array(-self.q)

    def _left_mul_matrix_hamilton(self) -> ArrayLike:
        return self.np.array(
            [
                [self.w, -self.z, self.y, self.x],
                [self.z, self.w, -self.x, self.y],
                [-self.y, self.x, self.w, self.z],
                [-self.x, -self.y, -self.z, self.w],
            ]
        )

    def _right_mul_matrix_hamilton(self) -> ArrayLike:
        return self.np.array(
            [
                [self.w, -self.z, self.y, self.x],
                [self.z, self.w, -self.x, self.y],
                [-self.y, self.x, self.w, -self.z],
                [-self.x, -self.y, self.z, self.w],
            ]
        )

    def left_mul_matrix(self) -> ArrayLike:
        if self.is_product_antimorphic:
            return self._left_mul_matrix_hamilton()
        return self._right_mul_matrix_hamilton()

    def right_mul_matrix(self) -> ArrayLike:
        if self.is_product_antimorphic:
            return self._right_mul_matrix_hamilton()
        return self._left_mul_matrix_hamilton()

    def __mul__(self, other: "Quaternion") -> Self:
        return self.__class__.from_array(
            self.np.tensordot(self.left_mul_matrix(), other.q, axes=(1, self._AXIS))
            # self.lib.einsum("ij...,j...->i...", self.left_mul_matrix(), other.q)
        )

    def __truediv__(self, other: "Quaternion") -> Self:
        return self * other.inverse()

    def exp(self) -> Self:
        norm_xyz = self.norm_xyz
        normalized_xyz = self.normalized_xyz
        return self.np.exp(self.w) * self.__class__.from_array_xyz(
            xyz=normalized_xyz * self.np.sin(norm_xyz), w=self.np.cos(norm_xyz)
        )

    def log(self) -> Self:
        norm_xyz = self.norm_xyz
        normalized_xyz = self.normalized_xyz
        theta = self.np.arctan2(norm_xyz, self.w)
        return self.__class__.from_array_xyz(
            xyz=normalized_xyz * theta, w=self.np.log(self.norm)
        )

    def __pow__(self, other: float) -> Self:
        normalized_xyz = self.normalized_xyz
        theta = self.np.arccos(self.w)
        return self.__class__.from_array_xyz(
            xyz=normalized_xyz * self.np.sin(other * theta),
            w=self.np.cos(other * theta),
        )

    def to_rotation_matrix(self, active: bool, cut: bool = True) -> ArrayLike:
        """
        The rotation matrix corresponding to the quaternion.

        Parameters
        ----------
        active : bool
            Whether to return an active (in terms of the object <==> the coordinate system is fixed)
            or passive (in terms of the coordinate system <==> the object is fixed) rotation matrix.
        cut : bool
            Whether to cut the last row and column of the result matrix
            to get a 3x3 matrix instead of a 4x4 matrix. Default is True.

        Returns
        -------
        ArrayLike
            M such that M x = q * x * q^-1 if active is True
            M such that M x = q^-1 * x * q if active is False

        """
        if active:
            res = self.np.tensordot(
                self.left_mul_matrix(), self.right_mul_matrix(), axes=(1, 0)
            )
            # res = self.lib.einsum(
            #     "ij...,j...->i...", self.left_mul_matrix(), self.right_mul_matrix()
            # )
        else:
            res = self.np.tensordot(
                self.right_mul_matrix(), self.left_mul_matrix(), axes=(1, 0)
            )
            # res = self.lib.einsum(
            #     "ij...,j...->i...", self.right_mul_matrix(), self.left_mul_matrix()
            # )

        if cut:
            return res[:3, :3]
        else:
            return res

    def rotate(self, vector: ArrayLike, active: bool, axis: int = 0) -> ArrayLike:
        return self.np.tensordot(
            self.to_rotation_matrix(active), vector, axes=(1, axis)
        ).moveaxis(0, axis)

    @classmethod
    def rotation_derivative_matrix(
        cls, omega: ArrayLike, active: bool, *, axis: int = 0
    ) -> ArrayLike:
        """
        The derivative of the rotation matrix corresponding to the quaternion.

        Parameters
        ----------
        omega : ArrayLike
            The angular velocity vector.

        Returns
        -------
        ArrayLike
            The derivative of the rotation matrix.

        """
        if omega.shape[0] != 3:
            raise ValueError("The angular velocity vector must have 3 components.")
        lib = _get_lib(omega)
        zero = lib.zeros_like(omega[0])
        o0 = omega.take(0, axis=0)
        o1 = omega.take(1, axis=0)
        o2 = omega.take(2, axis=0)
        if active:
            return lib.array(
                [
                    [zero, -o0, -o1, -o2],
                    [o0, zero, o2, -o1],
                    [o1, -o2, zero, o0],
                    [o2, o1, -o0, zero],
                ]
            )
        return lib.array(
            [
                [zero, -o0, -o1, -o2],
                [o0, zero, -o2, o1],
                [o1, o2, zero, -o0],
                [o2, -o1, o0, zero],
            ]
        )

    def rotation_derivative(
        self, omega: ArrayLike, active: bool, *, axis: int = 0
    ) -> Self:
        """
        The derivative of the rotation matrix corresponding to the quaternion.

        Parameters
        ----------
        omega : ArrayLike
            The angular velocity vector.

        Returns
        -------
        ArrayLike
            The derivative of the rotation matrix.

        """
        return self.__class__.from_array(
            self.np.tensordot(
                self.rotation_derivative_matrix(omega, active, axis=axis),
                self.q,
                (1, self._AXIS),
            )
        )

    def lerp(self, other: "Quaternion", t: ArrayLike, clamped: bool = False) -> Self:
        if clamped:
            t = self.np.clip(t, 0, 1)
        return self * (1 - t) + other * t

    def slerp(self, other: "Quaternion", t: ArrayLike, clamped: bool = False) -> Self:
        if clamped:
            t = self.np.clip(t, 0, 1)
        angle = self.angle(other)
        anglesin = self.np.sin(angle)
        tself = self.np.sin((1 - t) * angle) / anglesin
        tother = self.np.sin(t * angle) / anglesin
        return self * tself + other * tother

    @classmethod
    def from_to_rotation(
        cls, from_direction: ArrayLike, to_direction: ArrayLike, *, axis=0
    ) -> Self:
        lib = _get_lib(from_direction, to_direction)
        from_direction = from_direction / lib.linalg.norm(from_direction, axis=axis)
        to_direction = to_direction / lib.linalg.norm(to_direction, axis=axis)
        axis = lib.cross(from_direction, to_direction, axis=axis)
        angle = lib.arccos(lib.dot(from_direction, to_direction, axis=axis))
        return cls.from_angle_axis(angle, axis)

    @classmethod
    def look_rotation(cls, forward: ArrayLike, upwards: ArrayLike, *, axis=0) -> Self:
        lib = _get_lib(forward, upwards)
        dim = forward.ndim
        new_shape = lib.ones(dim, dtype=int)
        new_shape[axis] = -1
        rot_forward = cls.from_to_rotation(
            lib.array([0, 0, 1]).reshape(new_shape), forward
        )
        rot_upwards = cls.from_to_rotation(
            rot_forward.rotate(lib.array([0, 1, 0]), active=True).reshape(new_shape),
            upwards,
        )
        return rot_upwards * rot_forward

    def allclose(
        self, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs
    ) -> ArrayLike:
        return self._close("allclose", other, *args, as_rotation=as_rotation, **kwargs)

    def isclose(
        self, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs
    ) -> ArrayLike:
        return self._close("isclose", other, *args, as_rotation=as_rotation, **kwargs)

    def equal(
        self, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs
    ) -> ArrayLike:
        return self._close("equal", other, *args, as_rotation=as_rotation, **kwargs)

    def array_equal(
        self, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs
    ) -> ArrayLike:
        return self._close(
            "array_equal", other, *args, as_rotation=as_rotation, **kwargs
        )

    def __ne__(self, value: object) -> ArrayLike:
        return not self.__eq__(value)

    def __eq__(self, value: object) -> ArrayLike:
        if not isinstance(value, Quaternion):
            return False
        return self._close("equal", value, self.eq_as_rotation)

    def _close(
        self,
        name: str,
        other: "Quaternion",
        *args,
        as_rotation: bool | None = None,
        **kwargs,
    ) -> ArrayLike:
        if as_rotation is None:
            as_rotation = self.eq_as_rotation
        if as_rotation:
            return self.np.any(
                [
                    getattr(self.np, name)(self.q, other.q, *args, **kwargs),
                    getattr(self.np, name)(self.q, -other.q, *args, **kwargs),
                ]
            )
        return getattr(self.np, name)(self.q, other.q, *args, **kwargs)
