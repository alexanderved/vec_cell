//! `VecCell` is a `Vec` with interior mutability and dynamically checked borrow rules.
//! `VecCell` allows to take disjoint mutable borrows to its elements.
//!
//! # Example
//!
//! ```
//! use vec_cell::VecCell;
//!
//! let mut vec_cell: VecCell<i32> = VecCell::new();
//!
//! vec_cell.push(0);
//! vec_cell.push(1);
//! vec_cell.push(2);
//!
//! {
//!     assert_eq!(*vec_cell.borrow(0), 0);
//!     assert_eq!(*vec_cell.borrow(1), 1);
//!     assert_eq!(*vec_cell.borrow(2), 2);
//! }
//!
//! {
//!     let borrow_mut1 = &mut *vec_cell.borrow_mut(1);
//!     let borrow_mut2 = &mut *vec_cell.borrow_mut(2);
//!
//!     *borrow_mut1 = 10;
//!     *borrow_mut2 = 15;
//! }
//!
//! assert_eq!(vec_cell.pop(), Some(15));
//! assert_eq!(vec_cell.pop(), Some(10));
//! assert_eq!(vec_cell.pop(), Some(0));
//! ```

use std::cell::{Cell, UnsafeCell};
use std::convert::From;
use std::default::Default;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut, Drop};
use std::ptr::NonNull;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BorrowFlag {
    NotBorrowed,
    Reading(usize),
    Writing,
}

#[derive(Debug)]
struct BorrowRef<'borrow> {
    borrow_flag: &'borrow Cell<BorrowFlag>,
}

impl<'borrow> BorrowRef<'borrow> {
    fn new(borrow_flag: &'borrow Cell<BorrowFlag>) -> Option<Self> {
        let b = match borrow_flag.get() {
            BorrowFlag::NotBorrowed => BorrowFlag::Reading(1),
            BorrowFlag::Reading(n) => BorrowFlag::Reading(n + 1),
            BorrowFlag::Writing => return None,
        };
        borrow_flag.set(b);
        Some(Self { borrow_flag })
    }
}

impl Clone for BorrowRef<'_> {
    fn clone(&self) -> Self {
        match self.borrow_flag.get() {
            BorrowFlag::Reading(n) => self.borrow_flag.set(BorrowFlag::Reading(n + 1)),
            _ => unreachable!(),
        }

        Self {
            borrow_flag: self.borrow_flag,
        }
    }
}

impl Drop for BorrowRef<'_> {
    fn drop(&mut self) {
        let b = match self.borrow_flag.get() {
            BorrowFlag::Reading(n) if n > 1 => BorrowFlag::Reading(n - 1),
            _ => BorrowFlag::NotBorrowed,
        };

        self.borrow_flag.set(b);
    }
}

#[derive(Debug)]
struct BorrowRefMut<'borrow> {
    borrow_flag: &'borrow Cell<BorrowFlag>,
}

impl<'borrow> BorrowRefMut<'borrow> {
    fn new(borrow_flag: &'borrow Cell<BorrowFlag>) -> Option<Self> {
        match borrow_flag.get() {
            BorrowFlag::NotBorrowed => {
                borrow_flag.set(BorrowFlag::Writing);
                Some(Self { borrow_flag })
            }
            _ => None,
        }
    }
}

impl Drop for BorrowRefMut<'_> {
    fn drop(&mut self) {
        self.borrow_flag.set(BorrowFlag::NotBorrowed);
    }
}

/// A trait which is used to implement flattenning of nested types, e.g.
/// converting `Option<ElementRef<'_, Option<T>>>` to `Option<ElementRef<'_, T>>`.
pub trait Flatten {
    type Output;
    fn flatten(self) -> Self::Output;
}

/// A wrapper type for a immutably borrowed element from a `VecCell<T>`.
#[derive(Clone)]
pub struct ElementRef<'borrow, T: 'borrow> {
    value: NonNull<T>,
    #[allow(dead_code)]
    borrow_ref: BorrowRef<'borrow>,
}

impl<T: Debug> Debug for ElementRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", **self))
    }
}

impl<T> Deref for ElementRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: Until `ElementRef` is dropped, `BorrowRef` ensures that there won't be
        // any unique references which are aliasing with obtained shared references.
        unsafe { self.value.as_ref() }
    }
}

impl<'borrow, T> Flatten for Option<ElementRef<'borrow, Option<T>>> {
    type Output = Option<ElementRef<'borrow, T>>;

    /// Converts `Option<ElementRef<'_, Option<T>>>` to `Option<ElementRef<'_, T>>`.
    fn flatten(self) -> Self::Output {
        self.and_then(|element_ref_option| {
            match element_ref_option.as_ref() {
                Some(value) => Some(ElementRef {
                    value: unsafe { NonNull::new_unchecked(value as *const _ as *mut _) },
                    borrow_ref: element_ref_option.borrow_ref,
                }),
                None => None,
            }
        })
    }
}

/// A wrapper type for a mutably borrowed element from a `VecCell<T>`.
pub struct ElementRefMut<'borrow, T: 'borrow> {
    value: NonNull<T>,
    #[allow(dead_code)]
    borrow_ref_mut: BorrowRefMut<'borrow>,

    _p: PhantomData<&'borrow mut T>,
}

impl<T: Debug> Debug for ElementRefMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", **self))
    }
}

impl<T> Deref for ElementRefMut<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: Until `ElementRefMut` is dropped, `BorrowRefMut` ensures that
        // there won't be any references which are aliasing with obtained
        // shared references except shared references to VecCell.
        unsafe { self.value.as_ref() }
    }
}

impl<T> DerefMut for ElementRefMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Until `ElementRefMut` is dropped, `BorrowRefMut` ensures that
        // there won't be any references which are aliasing with obtained
        // unique reference except shared references to VecCell.
        unsafe { self.value.as_mut() }
    }
}

impl<'borrow, T> Flatten for Option<ElementRefMut<'borrow, Option<T>>> {
    type Output = Option<ElementRefMut<'borrow, T>>;

    /// Converts `Option<ElementRefMut<'_, Option<T>>>` to `Option<ElementRefMut<'_, T>>`.
    fn flatten(self) -> Self::Output {
        self.and_then(|mut element_ref_mut_option| {
            match element_ref_mut_option.as_mut() {
                Some(value) => Some(ElementRefMut {
                    value: unsafe { NonNull::new_unchecked(value as *mut _) },
                    borrow_ref_mut: element_ref_mut_option.borrow_ref_mut,

                    _p: PhantomData,
                }),
                None => None,
            }
        })
    }
}

/// An error which may occure after calling `VecCell::try_borrow` or `VecCell::try_borrow_mut`.
#[derive(Error, Debug)]
pub enum BorrowError {
    #[error("element is out of bounds")]
    ElementOutOfBounds,
    #[error("element is already borrowed mutably")]
    ElementAlreadyBorrowedMutably,
    #[error("element is already borrowed")]
    ElementAlreadyBorrowed,
}

/// A `Vec` with interior mutability and dynamically checked borrow rules
/// when interacting with its elements.
pub struct VecCell<T> {
    data: UnsafeCell<Vec<T>>,
    borrow_flags: Vec<Cell<BorrowFlag>>,

    len: usize,
}

impl<T> VecCell<T> {
    /// Creates a new empty `VecCell`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let vec_cell: VecCell<i32> = VecCell::new();
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: UnsafeCell::new(vec![]),
            borrow_flags: vec![],

            len: 0,
        }
    }

    /// Immutably borrows element with specified index.
    ///
    /// The borrow lasts until the returned `ElementRef` exits the scope.
    /// During this period any number of immutable borrows can be obtained but no mutable ones.
    ///
    /// # Panics
    /// Panics if the element is already borrowed mutably or `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    ///
    /// assert_eq!(*vec_cell.borrow(0), 0);
    /// assert_eq!(*vec_cell.borrow(1), 1);
    /// assert_eq!(*vec_cell.borrow(2), 2);
    /// ```
    #[inline]
    pub fn borrow(&self, index: usize) -> ElementRef<'_, T> {
        self.try_borrow(index)
            .unwrap_or_else(|err| panic!("Borrow error: {err}"))
    }

    /// Immutably borrows an element with specified index, returns an error if the element
    /// is already borrowed mutably or its index is out of bounds.
    ///
    /// The borrow lasts until the returned `ElementRef` exits the scope.
    /// During this period any number of immutable borrows can be obtained but no mutable ones.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let vec_cell: VecCell<i32> = VecCell::from(vec![0]);
    ///
    /// {
    ///     let borrow = vec_cell.try_borrow(0);
    ///     assert!(vec_cell.try_borrow(0).is_ok());
    /// }
    ///
    /// {
    ///     assert!(vec_cell.try_borrow(10).is_err());
    /// }
    ///
    /// {
    ///     let borrow_mut = vec_cell.try_borrow_mut(0);
    ///     assert!(vec_cell.try_borrow(0).is_err())
    /// }
    /// ```
    pub fn try_borrow(&self, index: usize) -> Result<ElementRef<'_, T>, BorrowError> {
        self.borrow_flags
            .get(index)
            .ok_or(BorrowError::ElementOutOfBounds)
            .and_then(|borrow_flag| {
                BorrowRef::new(borrow_flag).ok_or(BorrowError::ElementAlreadyBorrowedMutably)
            })
            .and_then(|borrow_ref| {
                (index < self.len)
                    .then(|| {
                        // SAFETY: Until `ElementRef` is dropped, `BorrowRef` ensures that we can't
                        // get any unique references which are aliasing with this pointer.
                        let element = unsafe { (*self.data.get()).as_mut_ptr().add(index) };

                        ElementRef {
                            // SAFETY: The pointer to the element is valid because:
                            //  1. The pointer to `Vec` which is obtained
                            //     from `UnsafeCell` is always valid;
                            //  2. The element is inside the bounds of `Vec` because `index < len`.
                            value: unsafe { NonNull::new_unchecked(element) },
                            borrow_ref,
                        }
                    })
                    .ok_or(BorrowError::ElementOutOfBounds)
            })
    }

    /// Mutably borrows element with specified index.
    ///
    /// The borrow lasts until the returned `ElementRefMut` exits the scope.
    /// During this period no borrows can be obtained.
    ///
    /// # Panics
    /// Panics if the element is already borrowed or `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let vec_cell: VecCell<i32> = VecCell::from(vec![0]);
    ///
    /// let mut borrow_mut = vec_cell.borrow_mut(0);
    /// *borrow_mut = 42;
    ///
    /// assert_eq!(*borrow_mut, 42);
    /// ```
    #[inline]
    pub fn borrow_mut(&self, index: usize) -> ElementRefMut<'_, T> {
        self.try_borrow_mut(index)
            .unwrap_or_else(|err| panic!("Mutable borrow error: {err}"))
    }

    /// Mutably borrows an element with specified index, returns an error if the element
    /// is already borrowed or its index is out of bounds.
    ///
    /// The borrow lasts until the returned `ElementRefMut` exits the scope.
    /// During this period no borrows can be obtained.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let vec_cell: VecCell<i32> = VecCell::from(vec![0]);
    ///
    /// {
    ///     assert!(vec_cell.try_borrow_mut(0).is_ok());
    ///     assert!(vec_cell.try_borrow_mut(10).is_err())
    /// }
    ///
    /// {
    ///     let borrow = vec_cell.try_borrow(0);
    ///     assert!(vec_cell.try_borrow_mut(0).is_err())
    /// }
    /// ```
    pub fn try_borrow_mut(&self, index: usize) -> Result<ElementRefMut<'_, T>, BorrowError> {
        self.borrow_flags
            .get(index)
            .ok_or(BorrowError::ElementOutOfBounds)
            .and_then(|borrow_flag| {
                BorrowRefMut::new(borrow_flag).ok_or(BorrowError::ElementAlreadyBorrowed)
            })
            .and_then(|borrow_ref_mut| {
                (index < self.len)
                    .then(|| {
                        // SAFETY: Until `ElementRefMut` is dropped, `BorrowRefMut` ensures
                        // that we can't get any references which are aliasing with this
                        // pointer except shared references to VecCell.
                        let element = unsafe { (*self.data.get()).as_mut_ptr().add(index) };

                        ElementRefMut {
                            // SAFETY: The pointer to the element is valid because:
                            //  1. The pointer to `Vec` which is obtained
                            //     from `UnsafeCell` is always valid;
                            //  2. The element is inside the bounds of `Vec` because `index < len`.
                            value: unsafe { NonNull::new_unchecked(element) },
                            borrow_ref_mut,

                            _p: PhantomData,
                        }
                    })
                    .ok_or(BorrowError::ElementOutOfBounds)
            })
    }

    /// Returns the number of elements in `VecCell`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    /// assert_eq!(vec_cell.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if `VecCell` contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::new();
    /// assert!(vec_cell.is_empty());
    ///
    /// vec_cell.push(0);
    /// assert!(!vec_cell.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an element to the back of a `VecCell`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::new();
    /// vec_cell.push(0);
    /// assert_eq!(*vec_cell.borrow(0), 0);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        self.data.get_mut().push(value);
        self.borrow_flags.push(Cell::new(BorrowFlag::NotBorrowed));

        self.len += 1;
    }

    /// Removes the last element from a `VecCell` and returns it, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    ///
    /// assert_eq!(vec_cell.pop(), Some(2));
    /// assert_eq!(vec_cell.pop(), Some(1));
    /// assert_eq!(vec_cell.pop(), Some(0));
    /// assert_eq!(vec_cell.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.borrow_flags.pop();
        self.data.get_mut().pop().map(|element| {
            self.len -= 1;

            element
        })
    }

    /// Move all elements from `other` to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell1: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    /// let mut vec_cell2: VecCell<i32> = VecCell::from(vec![3, 4, 5]);
    ///
    /// vec_cell1.append(&mut vec_cell2);
    ///
    /// for i in 0..6 {
    ///     assert_eq!(*vec_cell1.borrow(i), i as i32);
    /// }
    /// ```
    #[inline]
    pub fn append(&mut self, other: &mut VecCell<T>) {
        self.len += other.len();

        self.data.get_mut().append(other.data.get_mut());
        self.borrow_flags.append(&mut other.borrow_flags);
    }

    /// Move all elements from `vec` to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    /// let mut vec: Vec<i32> = vec![3, 4, 5];
    ///
    /// vec_cell.append_vec(&mut vec);
    ///
    /// for i in 0..6 {
    ///     assert_eq!(*vec_cell.borrow(i), i as i32);
    /// }
    /// ```
    #[inline]
    pub fn append_vec(&mut self, vec: &mut Vec<T>) {
        self.len += vec.len();

        self.borrow_flags
            .append(&mut vec![Cell::new(BorrowFlag::NotBorrowed); vec.len()]);
        self.data.get_mut().append(vec);
    }

    /// Insert an element at posiion `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    ///
    /// vec_cell.insert(1, 3);
    /// assert_eq!(*vec_cell.borrow(1), 3);
    /// ```
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        self.data.get_mut().insert(index, element);
        self.borrow_flags
            .insert(index, Cell::new(BorrowFlag::NotBorrowed));

        self.len += 1;
    }

    /// Removes and returns the element at position `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);
    ///
    /// let removed_element = vec_cell.remove(1);
    /// assert_eq!(removed_element, 1);
    /// ```
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        self.len -= 1;

        self.borrow_flags.remove(index);
        self.data.get_mut().remove(index)
    }

    /// Removes and returns the element at position `index`.
    ///
    /// The removed element is replaced by the last element.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2, 3, 4, 5]);
    ///
    /// let removed_element = vec_cell.swap_remove(1);
    /// assert_eq!(removed_element, 1);
    /// assert_eq!(*vec_cell.borrow(1), 5);
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.len -= 1;

        self.borrow_flags.swap_remove(index);
        self.data.get_mut().swap_remove(index)
    }

    /// Replaces and returns the element at position `index` with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or element at position `index` is already borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2, 3, 4, 5]);
    ///
    /// let replaced_element = vec_cell.replace(1, 10);
    /// assert_eq!(replaced_element, 1);
    /// assert_eq!(*vec_cell.borrow(1), 10);
    /// ```
    #[inline]
    pub fn replace(&self, index: usize, value: T) -> T {
        mem::replace(&mut *self.borrow_mut(index), value)
    }
}

impl<T: Default> VecCell<T> {
    /// Takes the element at position `index` and replaces it with `Default` value.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or element at position `index` is already borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2, 3, 4, 5]);
    ///
    /// let taken_element = vec_cell.take(1);
    /// assert_eq!(taken_element, 1);
    /// assert_eq!(*vec_cell.borrow(1), 0);
    /// ```
    #[inline]
    pub fn take(&self, index: usize) -> T {
        self.try_take(index)
            .unwrap_or_else(|err| panic!("Take error: {err}"))
    }

    /// Takes the element at position `index` and replaces it with `Default` value.
    ///
    /// If `index` is out of bounds or element at position `index`
    /// is already borrowed, returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_cell::VecCell;
    ///
    /// let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2, 3, 4, 5]);
    ///
    /// {
    ///     let taken_element = vec_cell.try_take(1);
    ///     assert_eq!(taken_element.unwrap(), 1);
    ///     assert_eq!(*vec_cell.borrow(1), 0);
    /// }
    ///
    /// {
    ///     let taken_element = vec_cell.try_take(10);
    ///     assert!(taken_element.is_err());
    /// }
    ///
    /// {
    ///     let borrow_mut = vec_cell.borrow_mut(2);
    ///     let taken_element = vec_cell.try_take(2);
    ///     assert!(taken_element.is_err());
    /// }
    /// ```
    #[inline]
    pub fn try_take(&self, index: usize) -> Result<T, BorrowError> {
        self.try_borrow_mut(index)
            .map(|mut element| mem::take(&mut *element))
    }
}

impl<T: Debug> Debug for VecCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum ElementDebug<'borrow, T> {
            Value(ElementRef<'borrow, T>),
            Borrowed,
        }

        impl<T: Debug> Debug for ElementDebug<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Self::Value(element_ref) => f.write_fmt(format_args!("{:?}", element_ref)),
                    Self::Borrowed => f.write_str("<borrowed>"),
                }
            }
        }

        f.debug_list()
            .entries(
                (0..self.len())
                    .into_iter()
                    .map(|i| match self.try_borrow(i) {
                        Ok(element_ref) => ElementDebug::Value(element_ref),
                        Err(_) => ElementDebug::Borrowed,
                    }),
            )
            .finish()
    }
}

impl<T> Default for VecCell<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> From<Vec<T>> for VecCell<T> {
    #[inline]
    fn from(data: Vec<T>) -> Self {
        let len = data.len();

        Self {
            data: UnsafeCell::new(data),
            borrow_flags: vec![Cell::new(BorrowFlag::NotBorrowed); len],

            len,
        }
    }
}

impl<T: Clone> From<&Vec<T>> for VecCell<T> {
    #[inline]
    fn from(data: &Vec<T>) -> Self {
        Self::from(data.clone())
    }
}

impl<T: Clone> From<&mut Vec<T>> for VecCell<T> {
    #[inline]
    fn from(data: &mut Vec<T>) -> Self {
        Self::from(data.clone())
    }
}

impl<T: Clone> From<&[T]> for VecCell<T> {
    #[inline]
    fn from(data: &[T]) -> Self {
        Self::from(data.to_vec())
    }
}

impl<T: Clone> From<&mut [T]> for VecCell<T> {
    #[inline]
    fn from(data: &mut [T]) -> Self {
        Self::from(data.to_vec())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_borrow() {
        let vec_cell: VecCell<i32> = VecCell::from(vec![0, 1]);

        let element0_borrow1 = &*vec_cell.borrow(0);
        let element0_borrow2 = &*vec_cell.borrow(0);

        assert_eq!(*element0_borrow1, 0);
        assert_eq!(*element0_borrow2, 0);
        assert_eq!(*element0_borrow1, *element0_borrow2);

        let element1_borrow1 = &*vec_cell.borrow(1);
        let element1_borrow2 = &*vec_cell.borrow(1);

        assert_eq!(*element1_borrow1, 1);
        assert_eq!(*element1_borrow2, 1);
        assert_eq!(*element1_borrow1, *element1_borrow2);
    }

    #[test]
    fn test_borrow_mut() {
        let vec_cell: VecCell<i32> = VecCell::from(vec![0, 1]);

        let element0_borrow_mut1 = &mut *vec_cell.borrow_mut(0);
        let element0_borrow_mut2 = vec_cell.try_borrow_mut(0);

        assert_eq!(*element0_borrow_mut1, 0);
        assert!(element0_borrow_mut2.is_err());

        let element1_borrow_mut1 = &mut *vec_cell.borrow_mut(1);
        let element1_borrow_mut2 = vec_cell.try_borrow_mut(1);

        assert_eq!(*element1_borrow_mut1, 1);
        assert!(element1_borrow_mut2.is_err());

        *element0_borrow_mut1 = 3;
        assert_eq!(*element0_borrow_mut1, 3);

        *element1_borrow_mut1 = 4;
        assert_eq!(*element1_borrow_mut1, 4);
    }

    #[test]
    fn test_borrow_and_borrow_mut() {
        let vec_cell: VecCell<i32> = VecCell::from(vec![0]);

        let borrow = &*vec_cell.borrow(0);
        assert_eq!(*borrow, 0);

        let borrow_mut = vec_cell.try_borrow_mut(0);
        assert!(borrow_mut.is_err());
    }

    #[test]
    fn test_borrow_mut_and_borrow() {
        let vec_cell: VecCell<i32> = VecCell::from(vec![0]);

        let mut element_ref_mut = vec_cell.borrow_mut(0);
        let borrow_mut = &mut *element_ref_mut;
        assert_eq!(*borrow_mut, 0);

        *borrow_mut = 1;
        assert_eq!(*borrow_mut, 1);

        let borrow = vec_cell.try_borrow(0);
        assert!(borrow.is_err());

        std::mem::drop(element_ref_mut);

        let borrow = &*vec_cell.borrow(0);
        assert_eq!(*borrow, 1);
    }

    #[test]
    fn test_len() {
        let mut vec_cell: VecCell<i32> = VecCell::new();
        assert!(vec_cell.is_empty());

        vec_cell.push(0);
        assert_eq!(vec_cell.len(), 1);

        {
            let borrow_mut = &mut *vec_cell.borrow_mut(0);
            let shared_ref = &vec_cell;

            *borrow_mut = 10;
            assert_eq!(shared_ref.len(), 1);
        }

        vec_cell.push(1);
        assert_eq!(vec_cell.len(), 2);
    }

    #[test]
    fn test_push() {
        let mut vec_cell: VecCell<i32> = VecCell::new();

        assert!(vec_cell.try_borrow(0).is_err());
        assert!(vec_cell.is_empty());

        vec_cell.push(0);
        vec_cell.push(1);
        vec_cell.push(2);

        assert_eq!(*vec_cell.borrow(0), 0);
        assert_eq!(*vec_cell.borrow(1), 1);
        assert_eq!(*vec_cell.borrow(2), 2);
        assert!(!vec_cell.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);

        assert_eq!(*vec_cell.borrow(0), 0);
        assert_eq!(*vec_cell.borrow(1), 1);
        assert_eq!(*vec_cell.borrow(2), 2);
        assert!(!vec_cell.is_empty());

        assert_eq!(vec_cell.pop(), Some(2));
        assert_eq!(vec_cell.pop(), Some(1));
        assert_eq!(vec_cell.pop(), Some(0));

        assert!(vec_cell.is_empty());
    }

    #[test]
    fn test_append() {
        let mut vec_cell1: VecCell<usize> = VecCell::from(vec![0, 1, 2]);
        let mut vec_cell2: VecCell<usize> = VecCell::from(vec![3, 4, 5]);

        vec_cell1.append(&mut vec_cell2);

        assert_eq!(vec_cell1.len(), 6);

        for i in 0..6 {
            assert_eq!(*vec_cell1.borrow(i), i);
        }
    }

    #[test]
    fn test_append_vec() {
        let mut vec_cell: VecCell<usize> = VecCell::from(vec![0, 1, 2]);
        let mut vec: Vec<usize> = vec![3, 4, 5];

        vec_cell.append_vec(&mut vec);

        assert_eq!(vec_cell.len(), 6);

        for i in 0..6 {
            assert_eq!(*vec_cell.borrow(i), i);
        }
    }

    #[test]
    fn test_insert() {
        let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2, 3, 4, 5]);

        assert_eq!(vec_cell.len(), 6);
        assert_eq!(*vec_cell.borrow(1), 1);

        vec_cell.insert(1, 6);
        assert_eq!(vec_cell.len(), 7);
        assert_eq!(*vec_cell.borrow(1), 6);

        assert_eq!(*vec_cell.borrow(4), 3);

        vec_cell.insert(4, 7);
        assert_eq!(vec_cell.len(), 8);
        assert_eq!(*vec_cell.borrow(4), 7);
    }

    #[test]
    fn test_remove() {
        let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);

        assert_eq!(vec_cell.len(), 3);

        assert_eq!(vec_cell.remove(0), 0);
        assert_eq!(vec_cell.len(), 2);

        assert_eq!(vec_cell.remove(1), 2);
        assert_eq!(vec_cell.len(), 1);

        assert_eq!(vec_cell.remove(0), 1);
        assert!(vec_cell.is_empty());
    }

    #[test]
    fn test_swap_remove() {
        let mut vec_cell: VecCell<i32> = VecCell::from(vec![0, 1, 2]);

        assert_eq!(vec_cell.len(), 3);

        assert_eq!(vec_cell.swap_remove(0), 0);
        assert_eq!(*vec_cell.borrow(0), 2);
        assert_eq!(vec_cell.len(), 2);

        assert_eq!(vec_cell.swap_remove(0), 2);
        assert_eq!(*vec_cell.borrow(0), 1);
        assert_eq!(vec_cell.len(), 1);

        assert_eq!(vec_cell.swap_remove(0), 1);
        assert!(vec_cell.is_empty());
    }

    #[test]
    fn test_element_ref_flatten() {
        let vec_cell: VecCell<Option<i32>> = VecCell::from(vec![Some(0), None]);

        {
            let borrow_option0_0 = vec_cell.try_borrow(0).ok();
            let borrow0_0 = borrow_option0_0.flatten();

            let borrow_option0_1 = vec_cell.try_borrow(0).ok();
            let borrow0_1 = borrow_option0_1.flatten();

            assert_eq!(**borrow0_0.as_ref().unwrap(), 0);
            assert_eq!(**borrow0_1.as_ref().unwrap(), 0);

            let borrow_option1 = vec_cell.try_borrow(1).ok();
            let borrow1 = borrow_option1.flatten();

            assert!(borrow1.is_none());

            let borrow_option2 = vec_cell.try_borrow(2).ok();
            let borrow2 = borrow_option2.flatten();

            assert!(borrow2.is_none());
        }

        {
            let mut borrow_mut0 = vec_cell.borrow_mut(0);
            *borrow_mut0 = Some(3);

            assert_eq!(*borrow_mut0, Some(3));
        }
    }

    #[test]
    fn test_element_ref_mut_flatten() {
        let vec_cell: VecCell<Option<i32>> = VecCell::from(vec![Some(0), None]);

        {
            let borrow_mut_option0_0 = vec_cell.try_borrow_mut(0).ok();
            let mut borrow_mut0_0 = borrow_mut_option0_0.flatten();

            let borrow_mut_option0_1 = vec_cell.try_borrow_mut(0).ok();
            let mut borrow_mut0_1 = borrow_mut_option0_1.flatten();

            if let Some(ref mut value_mut0_0) = borrow_mut0_0 {
                **value_mut0_0 = 5;
            }

            if let Some(ref mut value_mut0_1) = borrow_mut0_1 {
                **value_mut0_1 = 6;
            }

            assert_eq!(**borrow_mut0_0.as_ref().unwrap(), 5);
            assert!(borrow_mut0_1.as_ref().is_none());

            let borrow_mut_option1 = vec_cell.try_borrow_mut(1).ok();
            let mut borrow_mut1 = borrow_mut_option1.flatten();

            if let Some(ref mut value_mut1) = borrow_mut1 {
                **value_mut1 = 7;
            }

            assert!(borrow_mut1.is_none());

            let borrow_mut_option2 = vec_cell.try_borrow_mut(2).ok();
            let mut borrow_mut2 = borrow_mut_option2.flatten();

            if let Some(ref mut value_mut2) = borrow_mut2 {
                **value_mut2 = 8;
            }

            assert!(borrow_mut2.is_none());
        }

        {
            let borrow0 = vec_cell.borrow(0);

            assert_eq!(*borrow0, Some(5));
        }
    }
}
