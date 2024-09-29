import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Self


class Quaternion:
    _AXIS = 0

    def __init__(
        self,
        *,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        w: ArrayLike,
        is_product_antimorphic: bool = True,
        eq_as_rotation: bool = True
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
          use self.lerp(other, np.min(maxDegreesDelta / self.angle(other), 1)) instead
        

        References
        ----------
        Why and How to Avoid the Flipped Quaternion Multiplication
        https://arxiv.org/abs/1801.07478
        Handy Note for Quaternions
        https://www.mesw.co.jp/business/report/pdf/mss_18_07.pdf

        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.w = np.array(w)
        self.q = np.stack([self.x, self.y, self.z, self.w], axis=self._AXIS)
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
        return np.linalg.norm(self.q, axis=self._AXIS)

    @property
    def normalized(self) -> Self:
        return self / self.norm

    @property
    def norm_xyz(self) -> ArrayLike:
        return np.linalg.norm(self.xyz, axis=self._AXIS)

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

    def angle(self, other: "Quaternion" | None = None) -> float:
        if other is None:
            return 2 * np.arccos(self.w)
        return np.arccos(self.dot(other))
    
    def axis(self) -> ArrayLike:
        angle = self.angle()
        return self.q[:3] / np.sin(self.angle / 2)

    def to_angle_axis(self) -> tuple[ArrayLike, ArrayLike]:
        angle = self.angle()
        axis = self.q[:3] / np.sin(angle / 2)
        return angle, axis
    
    def from_angle_axis(cls, angle: ArrayLike, axis: ArrayLike) -> Self:
        anglehalf = angle / 2
        return cls.from_array_xyz(xyz=axis * np.sin(anglehalf), w=np.cos(anglehalf))

    def dot(self, other: "Quaternion") -> ArrayLike:
        return np.sum(self.q * other.q, axis=self._AXIS)

    @classmethod
    def from_euler(
        cls, x: ArrayLike, y: ArrayLike, z: ArrayLike, *, convention: str
    ) -> Self:
        pass

    def inverse(self) -> Self:
        return self.conjugate / np.sum(self.q**2)

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
        return self.__class__(x=-self.x, y=-self.y, z=-self.z, w=-self.w)

    def _left_mul_matrix_hamilton(self) -> ArrayLike:
        return np.array(
            [
                [self.w, -self.z, self.y, self.x],
                [self.z, self.w, -self.x, self.y],
                [-self.y, self.x, self.w, self.z],
                [-self.x, -self.y, -self.z, self.w],
            ]
        )

    def _right_mul_matrix_hamilton(self) -> ArrayLike:
        return np.array(
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
            np.tensordot(self.left_mul_matrix(), other.q, axes=(1, self._AXIS))
            # np.einsum("ij...,j...->i...", self.left_mul_matrix(), other.q)
        )

    def __truediv__(self, other: "Quaternion") -> Self:
        return self * other.inverse()

    def exp(self) -> Self:
        norm_xyz = self.norm_xyz
        normalized_xyz = self.normalized_xyz
        return np.exp(self.w) * self.__class__.from_array_xyz(
            xyz=normalized_xyz * np.sin(norm_xyz), w=np.cos(norm_xyz)
        )

    def log(self) -> Self:
        norm_xyz = self.norm_xyz
        normalized_xyz = self.normalized_xyz
        theta = np.arctan2(norm_xyz, self.w)
        return self.__class__.from_array_xyz(
            xyz=normalized_xyz * theta, w=np.log(self.norm)
        )

    def __pow__(self, other: float) -> Self:
        normalized_xyz = self.normalized_xyz
        theta = np.arccos(self.w)
        return self.__class__.from_array_xyz(
            xyz=normalized_xyz * np.sin(other * theta), w=np.cos(other * theta)
        )

    def rotation_matrix(self, active: bool, cut: bool = True) -> ArrayLike:
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
            res = np.tensordot(self.left_mul_matrix(), self.right_mul_matrix(), axes=(1, 0))
            # res = np.einsum(
            #     "ij...,j...->i...", self.left_mul_matrix(), self.right_mul_matrix()
            # )
        else:
            res = np.tensordot(self.right_mul_matrix(), self.left_mul_matrix(), axes=(1, 0))
            # res = np.einsum(
            #     "ij...,j...->i...", self.right_mul_matrix(), self.left_mul_matrix()
            # )

        if cut:
            return res[:3, :3]
        else:
            return res

    @classmethod
    def rotation_derivative_matrix(cls, omega: ArrayLike, active: bool) -> ArrayLike:
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
        zero = np.zeros_like(omega[0])
        if active:
            return np.array(
                [
                    [zero, -omega[0], -omega[1], -omega[2]],
                    [omega[0], zero, omega[2], -omega[1]],
                    [omega[1], -omega[2], zero, omega[0]],
                    [omega[2], omega[1], -omega[0], zero],
                ]
            )
        return np.array(
            [
                [zero, -omega[0], -omega[1], -omega[2]],
                [omega[0], zero, -omega[2], omega[1]],
                [omega[1], omega[2], zero, -omega[0]],
                [omega[2], -omega[1], omega[0], zero],
            ]
        )

    def rotation_derivative(self, omega: ArrayLike, active: bool) -> Quaternion:
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
            np.tensordot(
                self.rotation_derivative_matrix(omega, active),
                self.q,
                (1, self._AXIS),
            )
        )

    def lerp(self, other: "Quaternion", t: float, clamped: bool = False) -> Self:
        if clamped:
            t = np.clip(t, 0, 1)
        return self * (1 - t) + other * t
    
    def slerp(self, other: "Quaternion", t: float, clamped: bool = False) -> Self:
        if clamped:
            t = np.clip(t, 0, 1)
        angle = self.angle(other)
        anglesin = np.sin(angle)
        tself = np.sin((1 - t) * angle) / anglesin
        tother = np.sin(t * angle) / anglesin
        return self * tself + other * tother
    
    @classmethod
    def look_rotation(cls, forward: ArrayLike, upwards: ArrayLike) -> Self:
        forward = np.array(forward)
        upwards = np.array(upwards)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(upwards, forward)
        upwards = np.cross(forward, right)
        return cls.rotation_matrix(forward, upwards, right)

    def allclose(self, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs) -> ArrayLike:
        return self._close("allclose", other, *args, as_rotation=as_rotation, **kwargs)
    
    def isclose(self, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs) -> ArrayLike:
        return self._close("isclose", other, *args, as_rotation=as_rotation, **kwargs)
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Quaternion):
            return False
        return self._close("equal", value, self.eq_as_rotation)
    
    def _close(self, name: str, other: "Quaternion", *args, as_rotation: bool | None = None, **kwargs) -> ArrayLike:
        if as_rotation is None:
            as_rotation = self.eq_as_rotation
        if as_rotation:
            return np.any(
                [
                    getattr(np, name)(self.q, other.q, *args, **kwargs),
                    getattr(np, name)(self.q, -other.q, *args, **kwargs),
                ]
            )
        return getattr(np, name)(self.q, other.q, *args, **kwargs)