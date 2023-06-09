use crate::{ElementRef, ElementRefMut};

/// A trait which is used to implement flattenning of nested types, e.g.
/// converting `Option<ElementRef<'_, Option<T>>>` to `Option<ElementRef<'_, T>>`.
pub trait Flatten {
    type Output;
    fn flatten(self) -> Self::Output;
}

impl<'borrow, T> Flatten for Option<ElementRef<'borrow, Option<T>>> {
    type Output = Option<ElementRef<'borrow, T>>;

    /// Converts `Option<ElementRef<'_, Option<T>>>` to `Option<ElementRef<'_, T>>`.
    fn flatten(self) -> Self::Output {
        self.and_then(|element_ref_option| match element_ref_option.as_ref() {
            Some(value) => {
                // SAFETY: `value` is nonnull because it is obtained from
                // `ElementRef` which is guaranteed to be nonnull.
                Some(unsafe { ElementRef::new(value as *const T, element_ref_option.borrow_ref) })
            }
            None => None,
        })
    }
}

impl<'borrow, T> Flatten for Option<ElementRefMut<'borrow, Option<T>>> {
    type Output = Option<ElementRefMut<'borrow, T>>;

    /// Converts `Option<ElementRefMut<'_, Option<T>>>` to `Option<ElementRefMut<'_, T>>`.
    fn flatten(self) -> Self::Output {
        self.and_then(
            |mut element_ref_mut_option| match element_ref_mut_option.as_mut() {
                Some(value) => Some(
                    // SAFETY: `value` is nonnull because it is obtained from
                    // `ElementRefMut` which is guaranteed to be nonnull.
                    unsafe {
                        ElementRefMut::new(value as *mut T, element_ref_mut_option.borrow_ref_mut)
                    },
                ),
                None => None,
            },
        )
    }
}

impl<'borrow, T, E: From<String>> Flatten for Result<ElementRef<'borrow, Option<T>>, E> {
    type Output = Result<ElementRef<'borrow, T>, E>;

    /// Converts `Result<ElementRef<'_, Option<T>>, E>` to `Result<ElementRef<'_, T>, E>`.
    fn flatten(self) -> Self::Output {
        self.and_then(
            |element_ref_result_option| match element_ref_result_option.as_ref() {
                Some(value) => {
                    // SAFETY: `value` is nonnull because it is obtained from
                    // `ElementRef` which is guaranteed to be nonnull.
                    Ok(unsafe {
                        ElementRef::new(value as *const T, element_ref_result_option.borrow_ref)
                    })
                }
                None => Err("No value was found".to_string().into()),
            },
        )
    }
}

impl<'borrow, T, E: From<String>> Flatten for Result<ElementRefMut<'borrow, Option<T>>, E> {
    type Output = Result<ElementRefMut<'borrow, T>, E>;

    /// Converts `Result<ElementRefMut<'_, Option<T>>>` to `Result<ElementRefMut<'_, T>>`.
    fn flatten(self) -> Self::Output {
        self.and_then(|mut element_ref_mut_result_option| {
            match element_ref_mut_result_option.as_mut() {
                Some(value) => Ok(
                    // SAFETY: `value` is nonnull because it is obtained from
                    // `ElementRefMut` which is guaranteed to be nonnull.
                    unsafe {
                        ElementRefMut::new(
                            value as *mut T,
                            element_ref_mut_result_option.borrow_ref_mut,
                        )
                    },
                ),
                None => Err("No value was found".to_string().into()),
            }
        })
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn test_flatten_element_ref_option() {
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
    fn test_flatten_element_ref_mut_option() {
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

    #[test]
    fn test_flatten_element_ref_result_option() {
        let vec_cell: VecCell<Option<i32>> = VecCell::from(vec![Some(0), None]);

        {
            let borrow_option0_0 = vec_cell.try_borrow(0);
            let borrow0_0 = borrow_option0_0.flatten();

            let borrow_option0_1 = vec_cell.try_borrow(0);
            let borrow0_1 = borrow_option0_1.flatten();

            assert_eq!(**borrow0_0.as_ref().unwrap(), 0);
            assert_eq!(**borrow0_1.as_ref().unwrap(), 0);

            let borrow_option1 = vec_cell.try_borrow(1);
            let borrow1 = borrow_option1.flatten();

            assert!(borrow1.is_err());

            let borrow_option2 = vec_cell.try_borrow(2);
            let borrow2 = borrow_option2.flatten();

            assert!(borrow2.is_err());
        }

        {
            let mut borrow_mut0 = vec_cell.borrow_mut(0);
            *borrow_mut0 = Some(3);

            assert_eq!(*borrow_mut0, Some(3));
        }
    }

    #[test]
    fn test_flatten_element_ref_mut_result_option() {
        let vec_cell: VecCell<Option<i32>> = VecCell::from(vec![Some(0), None]);

        {
            let borrow_mut_option0_0 = vec_cell.try_borrow_mut(0);
            let mut borrow_mut0_0 = borrow_mut_option0_0.flatten();

            let borrow_mut_option0_1 = vec_cell.try_borrow_mut(0);
            let mut borrow_mut0_1 = borrow_mut_option0_1.flatten();

            if let Ok(ref mut value_mut0_0) = borrow_mut0_0 {
                **value_mut0_0 = 5;
            }

            if let Ok(ref mut value_mut0_1) = borrow_mut0_1 {
                **value_mut0_1 = 6;
            }

            assert_eq!(**borrow_mut0_0.as_ref().unwrap(), 5);
            assert!(borrow_mut0_1.as_ref().is_err());

            let borrow_mut_option1 = vec_cell.try_borrow_mut(1);
            let mut borrow_mut1 = borrow_mut_option1.flatten();

            if let Ok(ref mut value_mut1) = borrow_mut1 {
                **value_mut1 = 7;
            }

            assert!(borrow_mut1.is_err());

            let borrow_mut_option2 = vec_cell.try_borrow_mut(2);
            let mut borrow_mut2 = borrow_mut_option2.flatten();

            if let Ok(ref mut value_mut2) = borrow_mut2 {
                **value_mut2 = 8;
            }

            assert!(borrow_mut2.is_err());
        }

        {
            let borrow0 = vec_cell.borrow(0);

            assert_eq!(*borrow0, Some(5));
        }
    }
}
