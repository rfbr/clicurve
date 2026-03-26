/// Minimal TensorBoard protobuf definitions using prost derive macros.
/// No protoc or build.rs needed — these match the wire format of TensorBoard event files.
use prost::{Message, Oneof};

#[derive(Clone, PartialEq, Message)]
pub struct Event {
    #[prost(double, tag = "1")]
    pub wall_time: f64,
    #[prost(int64, tag = "2")]
    pub step: i64,
    #[prost(oneof = "event::What", tags = "3, 5")]
    pub what: Option<event::What>,
}

pub mod event {
    use super::*;

    #[derive(Clone, PartialEq, Oneof)]
    pub enum What {
        #[prost(string, tag = "3")]
        FileVersion(String),
        #[prost(message, tag = "5")]
        Summary(super::Summary),
    }
}

#[derive(Clone, PartialEq, Message)]
pub struct Summary {
    #[prost(message, repeated, tag = "1")]
    pub value: Vec<summary::Value>,
}

pub mod summary {
    use super::*;

    #[derive(Clone, PartialEq, Message)]
    pub struct Value {
        #[prost(string, tag = "1")]
        pub tag: String,
        #[prost(oneof = "value::Kind", tags = "2")]
        pub value: Option<value::Kind>,
    }

    pub mod value {
        use super::*;

        #[derive(Clone, PartialEq, Oneof)]
        pub enum Kind {
            #[prost(float, tag = "2")]
            SimpleValue(f32),
        }
    }
}
