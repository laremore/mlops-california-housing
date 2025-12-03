# scripts/build_and_run.ps1
Write-Host "=== Building Docker image ===" -ForegroundColor Green
docker build -t california-housing-ml .

Write-Host "`n=== Running Docker container ===" -ForegroundColor Green
docker run -d `
  -p 8000:8000 `
  --name housing-ml-service `
  california-housing-ml

Write-Host "`n=== Checking container status ===" -ForegroundColor Green
docker ps --filter "name=housing-ml-service"

Write-Host "`n=== Testing health endpoint ===" -ForegroundColor Green
Start-Sleep -Seconds 3
curl http://localhost:8000/health

Write-Host "`n=== Service is running! ===" -ForegroundColor Green
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Health check: http://localhost:8000/health" -ForegroundColor Yellow

Write-Host "`nCommands to manage container:" -ForegroundColor Cyan
Write-Host "  Stop: docker stop housing-ml-service"
Write-Host "  Start: docker start housing-ml-service"
Write-Host "  Remove: docker rm -f housing-ml-service"
Write-Host "  View logs: docker logs housing-ml-service"