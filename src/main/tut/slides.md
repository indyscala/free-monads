<!DOCTYPE html>
<html>
  <head>
    <title>Title</title>
    <meta charset="utf-8">
    <style>
      @import url(https://fonts.googleapis.com/css?family=Yanone+Kaffeesatz);
      @import url(https://fonts.googleapis.com/css?family=Droid+Serif:400,700,400italic);
      @import url(https://fonts.googleapis.com/css?family=Ubuntu+Mono:400,700,400italic);

      body { font-family: 'Droid Serif'; }
      h1, h2, h3 {
        font-family: 'Yanone Kaffeesatz';
        font-weight: normal;
      }
      .remark-code, .remark-inline-code { font-family: 'Ubuntu Mono'; }
    </style>
  </head>
  <body>
    <textarea id="source">

# Free Monads

## IndyScala, July 5, 2016

---

## Foreshadowing

I am going to give you a list of data now, and ask you to combine it into a single value later.

What data structure do you use?

---

## Detour: Monoids

```tut:silent
trait Monoid[A] {
  /** Also known as `x |+| y` */
  def combine(x: A, y: A): A

  def empty: A
}
```

--

### Three components

1. A type, `A`
2. A binary operation, `(A, A) => A`
3. An empty value of type `A`

--

### Three laws:

1. Left identity
2. Right identity
3. Associativity

```tut:invisible
import cats.{Monoid => _, _}
import cats.std.all._

object Monoid {
  def apply[A: Monoid] = implicitly[Monoid[A]]
}

implicit class MonoidOps[A](self: A) {
  def |+|(rhs: A)(implicit M: Monoid[A]) =
    M.combine(self, rhs)
}

implicit class AnyOps[A](self: A) {
  def <=>(rhs: A)(implicit A: Eq[A]) = {
    if (A.eqv(self, rhs))
      println(s"$self == $rhs")
    else
      sys.error(s"$self != $rhs")
  }
}
```

---

## `(String, +, "")` is a monoid

```tut:silent
implicit val StringMonoid = new Monoid[String] {
  def combine(x: String, y: String) = x + y
  val empty = ""
}
```

--

```tut
("x" |+| ("y" |+| "z")) <=> (("x" |+| "y") |+| "z")
(Monoid[String].empty |+| "x") <=> "x"
("x" |+| Monoid[String].empty) <=> "x"
```

---

## `(Int, +, 0)` is a monoid

```tut:silent
implicit val IntMonoid = new Monoid[Int] {
  def combine(x: Int, y: Int) = x + y
  val empty = 0
}
```

--

```tut
(3 |+| (4 |+| 5)) <=> ((3 |+| 4) |+| 5)
(Monoid[Int].empty |+| 3) <=> 3
(3 |+| Monoid[Int].empty) <=> 3
```

---

## `(List[A], :::, Nil)` is a monoid for all `A`

```tut:silent
implicit def ListMonoid[A] = new Monoid[List[A]] {
  def combine(x: List[A], y: List[A]) = x ::: y
  val empty = Nil
}
```

--

```tut
(List(3) |+| (List(4) |+| List(5))) <=> ((List(3) |+| List(4)) |+| List(5))
(Monoid[List[Int]].empty |+| List(3)) <=> List(3)
(List(3) |+| Monoid[List[Int]].empty) <=> List(3)
```

---

## Monoid laws

### Associativity

```tut:invisible
import org.scalacheck.Arbitrary
import org.scalacheck.Prop.forAll
```

```tut:silent
def associativity[A](implicit A: Arbitrary[A], M: Monoid[A]) =
  forAll { (x: A, y: A, z: A) =>
    ((x |+| y) |+| z) == (x |+| (y |+| z))
  }
```

--

```tut
associativity[String].check
associativity[Int].check
associativity[List[Int]].check
```

---

## Monoid laws

### Left identity

```tut:silent
def leftIdentity[A](implicit A: Arbitrary[A], M: Monoid[A]) =
  forAll { x: A =>
    (M.empty |+| x) == x
  }
```

--

```tut
leftIdentity[String].check
leftIdentity[Int].check
leftIdentity[List[Int]].check
```

---

## Monoid laws

### Right identity

```tut:silent
def rightIdentity[A](implicit A: Arbitrary[A], M: Monoid[A]) =
  forAll { x: A =>
    (x |+| M.empty) == x
  }
```

--

```tut
rightIdentity[String].check
rightIdentity[Int].check
rightIdentity[List[Int]].check
```

---

## Mapping between monoids

```tut
("foo".length + "bar".length) <=> ("foo" + "bar").length
"".length <=> 0
```

--

`String#length` a _monoid homomorphism_.

---

## Monoid homomorphism

```tut:invisible
import cats.laws._
import cats.laws.discipline._
```

```tut:silent
def monoidHomomorphismLaws[A: Arbitrary: Monoid, B: Monoid: Eq](f: A => B) = {
  forAll { (x: A, y: A) =>
    (f(x |+| y)) <-> (f(x) |+| f(y))
  }.check
  
  f(Monoid[A].empty) <=> Monoid[B].empty
}
```

--

```tut
monoidHomomorphismLaws[String, Int](_.length)
monoidHomomorphismLaws[List[Int], Int](_.sum)
```

---

## Free monoids

```tut:silent
trait FreeMonoid[F[_], A] extends Monoid[F[A]] {
  def foldMap[B](fa: F[A])(f: A => B)(implicit M: Monoid[B]): B
}
```

--

- Losslessly stores the structure of a monoid.

--

- If we can map the type, then we have a monoid homomorphism.

--

- Is _free_ to be re-interpreted as another monoid.

---

## Int is not a free monoid

```tut:fail
implicit val IntMonoid = new FreeMonoid[Int] {
  def combine(x: Int, y: Int) = x + y
  val empty = 0
  def foldMap[B](fa: Int)(f: Int => B)(implicit M: Monoid[B]) = ???
}
```

---

## `List[A]` is a free monoid

```tut:silent
implicit def ListMonoid[A] = new FreeMonoid[List, A] {
  def combine(x: List[A], y: List[A]) = x ::: y
  val empty = Nil
  def foldMap[B](fa: List[A])(f: A => B)(implicit M: Monoid[B]) =
    fa.foldLeft(M.empty)((acc, a) => M.combine(acc, f(a)))
}
```

--

```tut
val xs = List('1', '2', '3')
val ys = List('4', '5', '6')
def f(c: Char) = c - '0'
```

```tut:invisible
def foldMap[B: Monoid](fa: List[Char])(f: Char => B) = ListMonoid[Char].foldMap[B](fa)(f)
```

```tut
(foldMap(xs)(f) |+| foldMap(ys)(f)) <=> foldMap(xs |+| ys)(f)
(foldMap(List.empty)(f) <=> Monoid[Int].empty)
```

---

## A word from the marketing department

- Does it look more familiar if we call it this?

  `def mapReduce[B](mapper: A => B)(implicit reducer: Monoid[B]): B`

---

## What is a monad?

- "It's a collection."

--

  _Not really._

--

- "It's a type with one type parameter."

--

  _Necessary, but not sufficient._

--

- "It has a `flatMap`."

--

  _Almost always true, but neither necessary nor sufficient._

--

- "Something that works in a for-comprehension."

--

  _See `flatMap`._

--

- "A monad is just a monoid in the category of endofunctors. What's the problem?"

--

  _Um, now we thoroughly understand one of those words?_

---

## Monad

```tut:silent
trait Monad[F[_]] {
  def bind[A, B](fa: F[A])(f: A => F[B]): F[B]
  def pure[A](a: A): F[A]
}
```

--

### Three components

1. A type, `A`
2. A binary operation, `(A, A) => A`
3. An empty value of type `A`

--

### Three laws:

1. Left identity
2. Right identity
3. Associativity

```tut:invisible
object Monad {
  def apply[F[_]: Monad] = implicitly[Monad[F]]
}
```

---

## Option is a monad

```tut:invisible
def hash(s: String) = java.util.Base64.getEncoder.encodeToString(s.getBytes)
```

```tut:silent
implicit val OptionMonad = new Monad[Option] {
  def bind[A, B](fa: Option[A])(f: A => Option[B]) = fa flatMap f
  def pure[A](a: A) = Option(a)
}
```

--

```tut
def x = "s3cur3"
def f(password: String) = password match {
  case "password" => None
  case x => Some(hash(x))
}
def g(hashed: String) = if (hashed == hash("s3cur3")) Some("root") else None
```

```tut:invisible
import OptionMonad._
```

```tut
pure(x).flatMap(f) <=> f(x)
bind(Option(x))(pure) <=> Option(x)
bind(bind(Option(x))(f))(g) <=> bind(Option(x))(a => bind(f(a))(g))
```

---

## List is a monad

```tut:silent
implicit val ListMonad = new Monad[List] {
  def bind[A, B](fa: List[A])(f: A => List[B]) = fa flatMap f
  def pure[A](a: A) = List(a)
}
```

--

```tut
def x = 3
def f(n: Int) = (1 to n).toList
def g(n: Int) = List.fill(n)(n)
```

```tut:invisible
import ListMonad._
```

```tut

pure(x).flatMap(f) <=> f(x)
bind(List(x))(pure) <=> List(x)
bind(bind(List(x))(f))(g) <=> bind(List(x))(a => bind(f(a))(g))
```

---

## Task is a monad

```tut:invisible
import scalaz.concurrent.Task
def lookupZip(zip: Int) = Task.delay("Indianapolis")
def forecast(s: String) = Task.delay("too damn hot")
```

```tut:silent
import scalaz.concurrent.Task, Task.delay

implicit val TaskMonad = new Monad[Task] {
  def bind[A, B](fa: Task[A])(f: A => Task[B]) = fa flatMap f
  def pure[A](a: A) = delay(a)
}
```

--

```tut
def x = 46250
def f(zip: Int) = lookupZip(zip)
def g(city: String) = forecast(city)
```

```tut:invisible
import TaskMonad._

import org.scalacheck.Arbitrary.arbitrary
implicit def TaskArbitrary[A: Arbitrary]: Arbitrary[Task[A]] = {
  Arbitrary {
    for {
      a <- arbitrary[A]
    } yield Task.delay(a)
  }
}

import cats.kernel.Eq
implicit def TaskEq[A: Eq]: Eq[Task[A]] = Eq.by(_.run)
```

```tut
pure(x).flatMap(f).run <=> f(x).run
bind(delay(x))(pure).run <=> delay(x).run
bind(bind(delay(x))(f))(g).run <=> bind(delay(x))(a => bind(f(a))(g)).run
```

---

## Monad laws

### Left identity

```tut:silent
def leftIdentity[F[_], A, B](implicit A: Arbitrary[A], FAB: Arbitrary[A => F[B]], M: Monad[F], E: Eq[F[B]]) =
  forAll { (a: A, f: A => F[B]) =>
    M.bind(M.pure(a))(f) <-> f(a)
  }
```

--

```tut
leftIdentity[Option, Int, String].check
leftIdentity[List, String, Boolean].check
leftIdentity[Task, Boolean, Int].check
```

---

## Monad laws

### Right identiy

```tut:silent
def rightIdentity[F[_], A, B](implicit A: Arbitrary[F[A]], FAB: Arbitrary[A => F[B]], M: Monad[F], E: Eq[F[A]]) =
  forAll { fa: F[A] =>
    M.bind(fa)(M.pure) <-> fa
  }
```

--

```tut
rightIdentity[Option, Int, String].check
rightIdentity[List, String, Boolean].check
rightIdentity[Task, Boolean, Int].check
```

---

## Monad laws

### Associativity

```tut:silent
def associativity[F[_], A, B, C](implicit A: Arbitrary[F[A]], AB: Arbitrary[A => F[B]], BC: Arbitrary[B => F[C]], M: Monad[F], E: Eq[F[C]]) =
  forAll { (fa: F[A], f: A => F[B], g: B => F[C]) =>
    M.bind(M.bind(fa)(f))(g) <-> M.bind(fa)(a => M.bind(f(a))(g))    
  }
```

--

```tut
associativity[Option, String, Int, Long].check
associativity[List, Boolean, Float, Double].check
associativity[Task, String, Char, Byte].check
```

---

## Let's define some operations

```tut:silent
trait KVStoreOp[A]
case class Get(key: String) extends KVStoreOp[Option[String]]
case class Put(key: String, value: String) extends KVStoreOp[Unit]
```

---

## Write some boilerplate

```tut:silent
import cats.free._

type KVStore[A] = Free[KVStoreOp, A]

def get(key: String): KVStore[Option[String]] =
  Free.liftF[KVStoreOp, Option[String]](Get(key))

def put(key: String, value: String): KVStore[Unit] =
  Free.liftF[KVStoreOp, Unit](Put(key, value))
```

---

## `KVStore` is a monad

```tut:silent
def getOrElse[A](key: String, default: => String): KVStore[String] =
  for {
    a <- get(key)
  } yield a.getOrElse(default)

def incr(key: String): KVStore[Unit] =
  for {
    a <- getOrElse(key, "0")
    _ <- put(key, (a.toInt + 1).toString)
  } yield ()
```

---

## We can write monadic programs with KV store

```tut
def present(speaker: String) =
  for {
    _ <- put("last-speaker", speaker)
    _ <- incr(s"presentation-count/$speaker") 
  } yield ()

val p = for {
  _ <- present("Brad")
  _ <- present("Brad")
  _ <- present("Ross")
      s <- getOrElse("last-speaker", "")
  c <- getOrElse(s"presentation-count/$s", "0")
} yield (s, c.toInt)
```

---

## Natural transformation

```tut:silent
trait NaturalTransformation[F[_], G[_]] {
  def apply[A](fa: F[A]): G[A]
}
```

--

```tut:silent
val ListToVector = new NaturalTransformation[List, Vector] {
  def apply[A](fa: List[A]): Vector[A] = fa.toVector
}
```

```tut
ListToVector(List(1, 2, 3))
```

---

## Let's write an interpreter

```tut
import cats.{Id, ~>}
import scala.collection.mutable

val InMemory: KVStoreOp ~> Id  =
  new (KVStoreOp ~> Id) {
    val store = mutable.Map.empty[String, String]
    def apply[A](fa: KVStoreOp[A]): Id[A] =
      fa match {
        case Put(key, value) =>
          println(s"PUT $key $value")
          store(key) = value.toString
        case Get(key) =>
          println(s"GET $key")        
          store.get(key)
      }
  }
```

---

## Let's run our program

```tut
p.foldMap(InMemory)
```

---

## Let's write to consul

```tut:invisible
implicit val TaskMonad = new cats.Monad[Task] {
  def pure[A](a: A) = Task.delay(a)
  def flatMap[A, B](fa: Task[A])(f: A => Task[B]) = fa.flatMap(f)
}
```

```tut
import org.http4s._, client._, Uri._
val Consul: KVStoreOp ~> Task =
  new (KVStoreOp ~> Task) {
    val client = org.http4s.client.blaze.defaultClient
    def apply[A](fa: KVStoreOp[A]): Task[A] =
      fa match {
        case Put(key, value) =>
          client.fetch(
            Request(Method.PUT, uri("http://192.168.99.100:8500/v1/kv/")/key)
              .withBody(value.toString))(_ => Task.now((())))
        case Get(key) =>
          import _root_.argonaut._, Argonaut._, org.http4s.argonaut._
          client.expect[Json](
            Request(Method.GET, uri("http://192.168.99.100:8500/v1/kv/")/key))
            .map(_.arrayOrEmpty.head.fieldOrNull("Value").string.get)
            .map(s => Some(new String(java.util.Base64.getDecoder().decode(s.getBytes))).asInstanceOf[A])
      }
  }
p.foldMap(Consul).run
```

--

* - Note: this interpreter is terrible

---

    </textarea>
    <script src="https://gnab.github.io/remark/downloads/remark-latest.min.js">
    </script>
    <script>
      var slideshow = remark.create();
    </script>
  </body>
</html>


