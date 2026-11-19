//! Interfaces owned by the fixture crate. Two traits, each with one method.

pub trait Store {
    fn persist(&self, key: &str);
}

pub trait Render {
    fn render(&self) -> String;
}
