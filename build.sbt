tutSettings

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.scalacheck" %% "scalacheck" % "1.13.0",
  "org.scalaz" %% "scalaz-concurrent" % "7.1.8",
  "org.typelevel" %% "cats-free" % "0.6.0",
  "org.typelevel" %% "cats-laws" % "0.6.0",
  "org.http4s" %% "http4s-blaze-client" % "0.14.1",
  "org.http4s" %% "http4s-argonaut" % "0.14.1"    
)

