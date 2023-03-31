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