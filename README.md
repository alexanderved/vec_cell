# vec_cell
Rust Vec with interior mutability which allows to take disjoint mutable references to its elements.

```rust
use vec_cell::VecCell;

// Create `VecCell`.
let vec_cell: VecCell<i32> = VecCell::new();

// Push elements to `VecCell`.
vec_cell.push(0);
vec_cell.push(1);
vec_cell.push(2);

// Take immutable borrows to `VecCell` elements.
{
    assert_eq!(*vec_cell.borrow(0), 0);
    assert_eq!(*vec_cell.borrow(1), 1);
    assert_eq!(*vec_cell.borrow(2), 2);
}

// Take disjoint mutable borrows to `VecCell` elements.
{
    let borrow_mut1 = &mut *vec_cell.borrow_mut(1);
    let borrow_mut2 = &mut *vec_cell.borrow_mut(2);

    *borrow_mut1 = 10;
    *borrow_mut2 = 15;
}

// Pop elements from `VecCell`.
assert_eq!(vec_cell.pop(), 15);
assert_eq!(vec_cell.pop(), 10);
assert_eq!(vec_cell.pop(), 0);
```